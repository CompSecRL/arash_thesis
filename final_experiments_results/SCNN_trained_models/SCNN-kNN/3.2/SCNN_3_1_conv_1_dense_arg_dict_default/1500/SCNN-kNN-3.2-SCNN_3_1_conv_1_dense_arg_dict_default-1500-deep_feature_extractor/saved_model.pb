ик*
Хы
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
╛
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.6.22v2.6.1-9-gc2363d6d0258╞├&
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
shape:	└T*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	└T*
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
в
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
в
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
в
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

NoOpNoOp
т[
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Э[
valueУ[BР[ BЙ[
Ц
layer-0
layer_with_weights-0
layer-1
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
Д
layer-0
	layer-1

layer-2
layer-3
layer-4
layer-5
layer_with_weights-0
layer-6
layer_with_weights-1
layer-7
layer_with_weights-2
layer-8
layer_with_weights-3
layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer_with_weights-6
layer-23
 layer_with_weights-7
 layer-24
!layer-25
"regularization_losses
#trainable_variables
$	variables
%	keras_api
 
v
&0
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111
212
313
414
515
╢
&0
'1
(2
)3
*4
+5
,6
-7
68
79
.10
/11
812
913
014
115
:16
;17
218
319
420
521
<22
=23
н
>layer_metrics
regularization_losses
?layer_regularization_losses
trainable_variables
	variables
@non_trainable_variables

Alayers
Bmetrics
 
 
 
 
R
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
R
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
R
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
h

&kernel
'bias
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
h

(kernel
)bias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
h

*kernel
+bias
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
Ч
[axis
	,gamma
-beta
6moving_mean
7moving_variance
\regularization_losses
]trainable_variables
^	variables
_	keras_api
Ч
`axis
	.gamma
/beta
8moving_mean
9moving_variance
aregularization_losses
btrainable_variables
c	variables
d	keras_api
Ч
eaxis
	0gamma
1beta
:moving_mean
;moving_variance
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
R
jregularization_losses
ktrainable_variables
l	variables
m	keras_api
R
nregularization_losses
otrainable_variables
p	variables
q	keras_api
R
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
А	variables
Б	keras_api
V
Вregularization_losses
Гtrainable_variables
Д	variables
Е	keras_api
V
Жregularization_losses
Зtrainable_variables
И	variables
Й	keras_api
V
Кregularization_losses
Лtrainable_variables
М	variables
Н	keras_api
V
Оregularization_losses
Пtrainable_variables
Р	variables
С	keras_api
V
Тregularization_losses
Уtrainable_variables
Ф	variables
Х	keras_api
l

2kernel
3bias
Цregularization_losses
Чtrainable_variables
Ш	variables
Щ	keras_api
Ь
	Ъaxis
	4gamma
5beta
<moving_mean
=moving_variance
Ыregularization_losses
Ьtrainable_variables
Э	variables
Ю	keras_api
V
Яregularization_losses
аtrainable_variables
б	variables
в	keras_api
 
v
&0
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111
212
313
414
515
╢
&0
'1
(2
)3
*4
+5
,6
-7
68
79
.10
/11
812
913
014
115
:16
;17
218
319
420
521
<22
=23
▓
гlayer_metrics
"regularization_losses
 дlayer_regularization_losses
#trainable_variables
$	variables
еnon_trainable_variables
жlayers
зmetrics
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
60
71
82
93
:4
;5
<6
=7

0
1
 
 
 
 
▓
иmetrics
йlayer_metrics
Cregularization_losses
Dtrainable_variables
E	variables
кnon_trainable_variables
лlayers
 мlayer_regularization_losses
 
 
 
▓
нmetrics
оlayer_metrics
Gregularization_losses
Htrainable_variables
I	variables
пnon_trainable_variables
░layers
 ▒layer_regularization_losses
 
 
 
▓
▓metrics
│layer_metrics
Kregularization_losses
Ltrainable_variables
M	variables
┤non_trainable_variables
╡layers
 ╢layer_regularization_losses
 

&0
'1

&0
'1
▓
╖metrics
╕layer_metrics
Oregularization_losses
Ptrainable_variables
Q	variables
╣non_trainable_variables
║layers
 ╗layer_regularization_losses
 

(0
)1

(0
)1
▓
╝metrics
╜layer_metrics
Sregularization_losses
Ttrainable_variables
U	variables
╛non_trainable_variables
┐layers
 └layer_regularization_losses
 

*0
+1

*0
+1
▓
┴metrics
┬layer_metrics
Wregularization_losses
Xtrainable_variables
Y	variables
├non_trainable_variables
─layers
 ┼layer_regularization_losses
 
 

,0
-1

,0
-1
62
73
▓
╞metrics
╟layer_metrics
\regularization_losses
]trainable_variables
^	variables
╚non_trainable_variables
╔layers
 ╩layer_regularization_losses
 
 

.0
/1

.0
/1
82
93
▓
╦metrics
╠layer_metrics
aregularization_losses
btrainable_variables
c	variables
═non_trainable_variables
╬layers
 ╧layer_regularization_losses
 
 

00
11

00
11
:2
;3
▓
╨metrics
╤layer_metrics
fregularization_losses
gtrainable_variables
h	variables
╥non_trainable_variables
╙layers
 ╘layer_regularization_losses
 
 
 
▓
╒metrics
╓layer_metrics
jregularization_losses
ktrainable_variables
l	variables
╫non_trainable_variables
╪layers
 ┘layer_regularization_losses
 
 
 
▓
┌metrics
█layer_metrics
nregularization_losses
otrainable_variables
p	variables
▄non_trainable_variables
▌layers
 ▐layer_regularization_losses
 
 
 
▓
▀metrics
рlayer_metrics
rregularization_losses
strainable_variables
t	variables
сnon_trainable_variables
тlayers
 уlayer_regularization_losses
 
 
 
▓
фmetrics
хlayer_metrics
vregularization_losses
wtrainable_variables
x	variables
цnon_trainable_variables
чlayers
 шlayer_regularization_losses
 
 
 
▓
щmetrics
ъlayer_metrics
zregularization_losses
{trainable_variables
|	variables
ыnon_trainable_variables
ьlayers
 эlayer_regularization_losses
 
 
 
│
юmetrics
яlayer_metrics
~regularization_losses
trainable_variables
А	variables
Ёnon_trainable_variables
ёlayers
 Єlayer_regularization_losses
 
 
 
╡
єmetrics
Їlayer_metrics
Вregularization_losses
Гtrainable_variables
Д	variables
їnon_trainable_variables
Ўlayers
 ўlayer_regularization_losses
 
 
 
╡
°metrics
∙layer_metrics
Жregularization_losses
Зtrainable_variables
И	variables
·non_trainable_variables
√layers
 №layer_regularization_losses
 
 
 
╡
¤metrics
■layer_metrics
Кregularization_losses
Лtrainable_variables
М	variables
 non_trainable_variables
Аlayers
 Бlayer_regularization_losses
 
 
 
╡
Вmetrics
Гlayer_metrics
Оregularization_losses
Пtrainable_variables
Р	variables
Дnon_trainable_variables
Еlayers
 Жlayer_regularization_losses
 
 
 
╡
Зmetrics
Иlayer_metrics
Тregularization_losses
Уtrainable_variables
Ф	variables
Йnon_trainable_variables
Кlayers
 Лlayer_regularization_losses
 

20
31

20
31
╡
Мmetrics
Нlayer_metrics
Цregularization_losses
Чtrainable_variables
Ш	variables
Оnon_trainable_variables
Пlayers
 Рlayer_regularization_losses
 
 

40
51

40
51
<2
=3
╡
Сmetrics
Тlayer_metrics
Ыregularization_losses
Ьtrainable_variables
Э	variables
Уnon_trainable_variables
Фlayers
 Хlayer_regularization_losses
 
 
 
╡
Цmetrics
Чlayer_metrics
Яregularization_losses
аtrainable_variables
б	variables
Шnon_trainable_variables
Щlayers
 Ъlayer_regularization_losses
 
 
8
60
71
82
93
:4
;5
<6
=7
╞
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
22
23
 24
!25
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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

80
91
 
 
 
 

:0
;1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
<0
=1
 
 
 
 
 
 
 
И
serving_default_left_inputsPlaceholder*,
_output_shapes
:         ▄*
dtype0*!
shape:         ▄
Ч
StatefulPartitionedCallStatefulPartitionedCallserving_default_left_inputsstream_2_conv_1/kernelstream_2_conv_1/biasstream_1_conv_1/kernelstream_1_conv_1/biasstream_0_conv_1/kernelstream_0_conv_1/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/beta%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/beta#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense_1/kerneldense_1/bias%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/beta*$
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
GPU2*0J 8В */
f*R(
&__inference_signature_wrapper_13980627
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
─
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*stream_0_conv_1/kernel/Read/ReadVariableOp(stream_0_conv_1/bias/Read/ReadVariableOp*stream_1_conv_1/kernel/Read/ReadVariableOp(stream_1_conv_1/bias/Read/ReadVariableOp*stream_2_conv_1/kernel/Read/ReadVariableOp(stream_2_conv_1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOpConst*%
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
GPU2*0J 8В **
f%R#
!__inference__traced_save_13982805
▀
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamestream_0_conv_1/kernelstream_0_conv_1/biasstream_1_conv_1/kernelstream_1_conv_1/biasstream_2_conv_1/kernelstream_2_conv_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization_1/gammabatch_normalization_1/betabatch_normalization_2/gammabatch_normalization_2/betadense_1/kerneldense_1/biasbatch_normalization_3/gammabatch_normalization_3/betabatch_normalization/moving_mean#batch_normalization/moving_variance!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance*$
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
GPU2*0J 8В *-
f(R&
$__inference__traced_restore_13982887ЮХ%
╕+
ъ
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_13978108

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
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
moments/StopGradient▒
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                  @2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
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
╫#<2
AssignMovingAvg/decayд
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
╫#<2
AssignMovingAvg_1/decayк
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
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
 :                  @2
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
 :                  @2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                  @2

IdentityЄ
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
╠
┐
__inference_loss_fn_0_13982677T
>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource:@
identityИв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpё
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absй
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
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
╝
r
V__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_13978520

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
╣Ц
т
G__inference_basemodel_layer_call_and_return_conditional_losses_13979925
inputs_0
inputs_1
inputs_2.
stream_2_conv_1_13979832:@&
stream_2_conv_1_13979834:@.
stream_1_conv_1_13979837:@&
stream_1_conv_1_13979839:@.
stream_0_conv_1_13979842:@&
stream_0_conv_1_13979844:@,
batch_normalization_2_13979847:@,
batch_normalization_2_13979849:@,
batch_normalization_2_13979851:@,
batch_normalization_2_13979853:@,
batch_normalization_1_13979856:@,
batch_normalization_1_13979858:@,
batch_normalization_1_13979860:@,
batch_normalization_1_13979862:@*
batch_normalization_13979865:@*
batch_normalization_13979867:@*
batch_normalization_13979869:@*
batch_normalization_13979871:@#
dense_1_13979885:	└T
dense_1_13979887:T,
batch_normalization_3_13979890:T,
batch_normalization_3_13979892:T,
batch_normalization_3_13979894:T,
batch_normalization_3_13979896:T
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв'stream_0_conv_1/StatefulPartitionedCallв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв'stream_1_conv_1/StatefulPartitionedCallв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв'stream_2_conv_1/StatefulPartitionedCallв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpБ
#stream_2_input_drop/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_139787602%
#stream_2_input_drop/PartitionedCallБ
#stream_1_input_drop/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_139787672%
#stream_1_input_drop/PartitionedCallБ
#stream_0_input_drop/PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_139787742%
#stream_0_input_drop/PartitionedCallы
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_2_input_drop/PartitionedCall:output:0stream_2_conv_1_13979832stream_2_conv_1_13979834*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_139787972)
'stream_2_conv_1/StatefulPartitionedCallы
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_1_input_drop/PartitionedCall:output:0stream_1_conv_1_13979837stream_1_conv_1_13979839*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_139788242)
'stream_1_conv_1/StatefulPartitionedCallы
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_13979842stream_0_conv_1_13979844*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_139788512)
'stream_0_conv_1/StatefulPartitionedCall╤
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_13979847batch_normalization_2_13979849batch_normalization_2_13979851batch_normalization_2_13979853*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_139788762/
-batch_normalization_2/StatefulPartitionedCall╤
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_13979856batch_normalization_1_13979858batch_normalization_1_13979860batch_normalization_1_13979862*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_139789052/
-batch_normalization_1/StatefulPartitionedCall├
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_13979865batch_normalization_13979867batch_normalization_13979869batch_normalization_13979871*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_139789342-
+batch_normalization/StatefulPartitionedCallЪ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_activation_2_layer_call_and_return_conditional_losses_139789492
activation_2/PartitionedCallЪ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_activation_1_layer_call_and_return_conditional_losses_139789562
activation_1/PartitionedCallТ
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_139789632
activation/PartitionedCallТ
stream_2_drop_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_139789702!
stream_2_drop_1/PartitionedCallТ
stream_1_drop_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_139789772!
stream_1_drop_1/PartitionedCallР
stream_0_drop_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_139789842!
stream_0_drop_1/PartitionedCallл
(global_average_pooling1d/PartitionedCallPartitionedCall(stream_0_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_139789912*
(global_average_pooling1d/PartitionedCall▒
*global_average_pooling1d_1/PartitionedCallPartitionedCall(stream_1_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *a
f\RZ
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_139789982,
*global_average_pooling1d_1/PartitionedCall▒
*global_average_pooling1d_2/PartitionedCallPartitionedCall(stream_2_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *a
f\RZ
X__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_139790052,
*global_average_pooling1d_2/PartitionedCall·
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:03global_average_pooling1d_1/PartitionedCall:output:03global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_139790152
concatenate/PartitionedCallН
dense_1_dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_dense_1_dropout_layer_call_and_return_conditional_losses_139790222!
dense_1_dropout/PartitionedCall║
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_13979885dense_1_13979887*
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
GPU2*0J 8В *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_139790402!
dense_1/StatefulPartitionedCall─
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_13979890batch_normalization_3_13979892batch_normalization_3_13979894batch_normalization_3_13979896*
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
GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_139786062/
-batch_normalization_3/StatefulPartitionedCallз
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
GPU2*0J 8В *Y
fTRR
P__inference_dense_activation_1_layer_call_and_return_conditional_losses_139790602$
"dense_activation_1/PartitionedCall╦
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_13979842*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absй
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul╤
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_13979837*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareй
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const┌
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul╦
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_13979832*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absй
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul░
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_13979885*
_output_shapes
:	└T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└T2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const╖
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЖ
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

IdentityЗ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_1_conv_1/StatefulPartitionedCall9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         ▄:         ▄:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:         ▄
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:         ▄
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:         ▄
"
_user_specified_name
inputs_2
Т
o
Q__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_13981734

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ▄2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ▄2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
О
░
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_13981980

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2
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
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▄@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
Т
o
Q__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_13978767

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ▄2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ▄2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
Ў
f
J__inference_activation_2_layer_call_and_return_conditional_losses_13982364

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         ▄@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         ▄@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
х
N
2__inference_stream_1_drop_1_layer_call_fn_13982396

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_139789772
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ▄@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
ў
l
M__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_13982391

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ▄@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ▄@*
dtype0*
seed╖*
seed2╖2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ▄@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
╕
Г
,__inference_basemodel_layer_call_fn_13981264
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

unknown_17:	└T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityИвStatefulPartitionedCall╡
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
:         T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_basemodel_layer_call_and_return_conditional_losses_139797182
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
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         ▄:         ▄:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:         ▄
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:         ▄
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:         ▄
"
_user_specified_name
inputs/2
▀
K
/__inference_activation_1_layer_call_fn_13982349

inputs
identity╨
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_activation_1_layer_call_and_return_conditional_losses_139789562
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ▄@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
е
Y
=__inference_global_average_pooling1d_2_layer_call_fn_13982494

inputs
identityт
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
GPU2*0J 8В *a
f\RZ
X__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_139785682
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
И
r
V__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_13978991

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
:         @2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
■
k
M__inference_dense_1_dropout_layer_call_and_return_conditional_losses_13982541

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         └2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         └2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         └:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
О
k
M__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_13978977

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ▄@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ▄@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
Й+
ь
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13979448

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
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
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         ▄@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
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
╫#<2
AssignMovingAvg/decayд
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
╫#<2
AssignMovingAvg_1/decayк
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2
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
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ▄@2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▄@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
М
i
M__inference_dense_1_dropout_layer_call_and_return_conditional_losses_13979166

inputs
identity[
IdentityIdentityinputs*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         └:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
ЇЬ
╥
E__inference_model_1_layer_call_and_return_conditional_losses_13980883

inputs[
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
0basemodel_dense_1_matmul_readvariableop_resource:	└T?
1basemodel_dense_1_biasadd_readvariableop_resource:TO
Abasemodel_batch_normalization_3_batchnorm_readvariableop_resource:TS
Ebasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:TQ
Cbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource:TQ
Cbasemodel_batch_normalization_3_batchnorm_readvariableop_2_resource:T
identityИв6basemodel/batch_normalization/batchnorm/ReadVariableOpв8basemodel/batch_normalization/batchnorm/ReadVariableOp_1в8basemodel/batch_normalization/batchnorm/ReadVariableOp_2в:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpв8basemodel/batch_normalization_1/batchnorm/ReadVariableOpв:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1в:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2в<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpв8basemodel/batch_normalization_2/batchnorm/ReadVariableOpв:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1в:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2в<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpв8basemodel/batch_normalization_3/batchnorm/ReadVariableOpв:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1в:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2в<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpв(basemodel/dense_1/BiasAdd/ReadVariableOpв'basemodel/dense_1/MatMul/ReadVariableOpв0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpв<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpв0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpв<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpв0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpв<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЫ
&basemodel/stream_2_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:         ▄2(
&basemodel/stream_2_input_drop/IdentityЫ
&basemodel/stream_1_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:         ▄2(
&basemodel/stream_1_input_drop/IdentityЫ
&basemodel/stream_0_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:         ▄2(
&basemodel/stream_0_input_drop/Identityн
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        21
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimО
+basemodel/stream_2_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_2_input_drop/Identity:output:08basemodel/stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄2-
+basemodel/stream_2_conv_1/conv1d/ExpandDimsЖ
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpи
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
-basemodel/stream_2_conv_1/conv1d/ExpandDims_1Я
 basemodel/stream_2_conv_1/conv1dConv2D4basemodel/stream_2_conv_1/conv1d/ExpandDims:output:06basemodel/stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2"
 basemodel/stream_2_conv_1/conv1dс
(basemodel/stream_2_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        2*
(basemodel/stream_2_conv_1/conv1d/Squeeze┌
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpї
!basemodel/stream_2_conv_1/BiasAddBiasAdd1basemodel/stream_2_conv_1/conv1d/Squeeze:output:08basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2#
!basemodel/stream_2_conv_1/BiasAddн
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        21
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimО
+basemodel/stream_1_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_1_input_drop/Identity:output:08basemodel/stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄2-
+basemodel/stream_1_conv_1/conv1d/ExpandDimsЖ
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpи
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
-basemodel/stream_1_conv_1/conv1d/ExpandDims_1Я
 basemodel/stream_1_conv_1/conv1dConv2D4basemodel/stream_1_conv_1/conv1d/ExpandDims:output:06basemodel/stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2"
 basemodel/stream_1_conv_1/conv1dс
(basemodel/stream_1_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        2*
(basemodel/stream_1_conv_1/conv1d/Squeeze┌
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpї
!basemodel/stream_1_conv_1/BiasAddBiasAdd1basemodel/stream_1_conv_1/conv1d/Squeeze:output:08basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2#
!basemodel/stream_1_conv_1/BiasAddн
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        21
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimО
+basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_0_input_drop/Identity:output:08basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄2-
+basemodel/stream_0_conv_1/conv1d/ExpandDimsЖ
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpи
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
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1Я
 basemodel/stream_0_conv_1/conv1dConv2D4basemodel/stream_0_conv_1/conv1d/ExpandDims:output:06basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1dс
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        2*
(basemodel/stream_0_conv_1/conv1d/Squeeze┌
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpї
!basemodel/stream_0_conv_1/BiasAddBiasAdd1basemodel/stream_0_conv_1/conv1d/Squeeze:output:08basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2#
!basemodel/stream_0_conv_1/BiasAddЄ
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpз
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
-basemodel/batch_normalization_2/batchnorm/add├
/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm/Rsqrt■
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpЕ
-basemodel/batch_normalization_2/batchnorm/mulMul3basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/mul 
/basemodel/batch_normalization_2/batchnorm/mul_1Mul*basemodel/stream_2_conv_1/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@21
/basemodel/batch_normalization_2/batchnorm/mul_1°
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1Е
/basemodel/batch_normalization_2/batchnorm/mul_2MulBbasemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm/mul_2°
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2Г
-basemodel/batch_normalization_2/batchnorm/subSubBbasemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/subК
/basemodel/batch_normalization_2/batchnorm/add_1AddV23basemodel/batch_normalization_2/batchnorm/mul_1:z:01basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@21
/basemodel/batch_normalization_2/batchnorm/add_1Є
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpз
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
-basemodel/batch_normalization_1/batchnorm/add├
/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm/Rsqrt■
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpЕ
-basemodel/batch_normalization_1/batchnorm/mulMul3basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/mul 
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_1_conv_1/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@21
/basemodel/batch_normalization_1/batchnorm/mul_1°
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1Е
/basemodel/batch_normalization_1/batchnorm/mul_2MulBbasemodel/batch_normalization_1/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm/mul_2°
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2Г
-basemodel/batch_normalization_1/batchnorm/subSubBbasemodel/batch_normalization_1/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/subК
/basemodel/batch_normalization_1/batchnorm/add_1AddV23basemodel/batch_normalization_1/batchnorm/mul_1:z:01basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@21
/basemodel/batch_normalization_1/batchnorm/add_1ь
6basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype028
6basemodel/batch_normalization/batchnorm/ReadVariableOpг
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
+basemodel/batch_normalization/batchnorm/add╜
-basemodel/batch_normalization/batchnorm/RsqrtRsqrt/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/Rsqrt°
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp¤
+basemodel/batch_normalization/batchnorm/mulMul1basemodel/batch_normalization/batchnorm/Rsqrt:y:0Bbasemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/mul∙
-basemodel/batch_normalization/batchnorm/mul_1Mul*basemodel/stream_0_conv_1/BiasAdd:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2/
-basemodel/batch_normalization/batchnorm/mul_1Є
8basemodel/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization/batchnorm/ReadVariableOp_1¤
-basemodel/batch_normalization/batchnorm/mul_2Mul@basemodel/batch_normalization/batchnorm/ReadVariableOp_1:value:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/mul_2Є
8basemodel/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization/batchnorm/ReadVariableOp_2√
+basemodel/batch_normalization/batchnorm/subSub@basemodel/batch_normalization/batchnorm/ReadVariableOp_2:value:01basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/subВ
-basemodel/batch_normalization/batchnorm/add_1AddV21basemodel/batch_normalization/batchnorm/mul_1:z:0/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2/
-basemodel/batch_normalization/batchnorm/add_1о
basemodel/activation_2/ReluRelu3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ▄@2
basemodel/activation_2/Reluо
basemodel/activation_1/ReluRelu3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ▄@2
basemodel/activation_1/Reluи
basemodel/activation/ReluRelu1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ▄@2
basemodel/activation/Relu╢
"basemodel/stream_2_drop_1/IdentityIdentity)basemodel/activation_2/Relu:activations:0*
T0*,
_output_shapes
:         ▄@2$
"basemodel/stream_2_drop_1/Identity╢
"basemodel/stream_1_drop_1/IdentityIdentity)basemodel/activation_1/Relu:activations:0*
T0*,
_output_shapes
:         ▄@2$
"basemodel/stream_1_drop_1/Identity┤
"basemodel/stream_0_drop_1/IdentityIdentity'basemodel/activation/Relu:activations:0*
T0*,
_output_shapes
:         ▄@2$
"basemodel/stream_0_drop_1/Identity╕
9basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9basemodel/global_average_pooling1d/Mean/reduction_indices¤
'basemodel/global_average_pooling1d/MeanMean+basemodel/stream_0_drop_1/Identity:output:0Bbasemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2)
'basemodel/global_average_pooling1d/Mean╝
;basemodel/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d_1/Mean/reduction_indicesГ
)basemodel/global_average_pooling1d_1/MeanMean+basemodel/stream_1_drop_1/Identity:output:0Dbasemodel/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2+
)basemodel/global_average_pooling1d_1/Mean╝
;basemodel/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d_2/Mean/reduction_indicesГ
)basemodel/global_average_pooling1d_2/MeanMean+basemodel/stream_2_drop_1/Identity:output:0Dbasemodel/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2+
)basemodel/global_average_pooling1d_2/MeanИ
!basemodel/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!basemodel/concatenate/concat/axis╩
basemodel/concatenate/concatConcatV20basemodel/global_average_pooling1d/Mean:output:02basemodel/global_average_pooling1d_1/Mean:output:02basemodel/global_average_pooling1d_2/Mean:output:0*basemodel/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:         └2
basemodel/concatenate/concatо
"basemodel/dense_1_dropout/IdentityIdentity%basemodel/concatenate/concat:output:0*
T0*(
_output_shapes
:         └2$
"basemodel/dense_1_dropout/Identity─
'basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	└T*
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
basemodel/dense_1/BiasAddЄ
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02:
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpз
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
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpЕ
-basemodel/batch_normalization_3/batchnorm/mulMul3basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/mulЄ
/basemodel/batch_normalization_3/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T21
/basemodel/batch_normalization_3/batchnorm/mul_1°
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1Е
/basemodel/batch_normalization_3/batchnorm/mul_2MulBbasemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm/mul_2°
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
:         T21
/basemodel/batch_normalization_3/batchnorm/add_1╛
$basemodel/dense_activation_1/SigmoidSigmoid3basemodel/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2&
$basemodel/dense_activation_1/Sigmoid°
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absй
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul■
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareй
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const┌
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul°
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absй
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul╨
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	└T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└T2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const╖
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulГ
IdentityIdentity(basemodel/dense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         T2

IdentityЬ
NoOpNoOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp9^basemodel/batch_normalization/batchnorm/ReadVariableOp_19^basemodel/batch_normalization/batchnorm/ReadVariableOp_2;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_3/batchnorm/ReadVariableOp;^basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 2p
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
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp2|
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp2|
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
╣
▓
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_13978210

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
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
 :                  @2
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
√
p
Q__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_13979509

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ▄2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ▄*
dtype0*
seed╖*
seed2╖2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ▄2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
Я
г
2__inference_stream_0_conv_1_layer_call_fn_13981761

inputs
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_139788512
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▄: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
О
k
M__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_13978984

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ▄@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ▄@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
ї
з
E__inference_dense_1_layer_call_and_return_conditional_losses_13982576

inputs1
matmul_readvariableop_resource:	└T-
biasadd_readvariableop_resource:T
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└T*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2	
BiasAdd╛
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└T2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const╖
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
╫#<2"
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

Identityп
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
ё
╙
8__inference_batch_normalization_1_layer_call_fn_13982066

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_139793882
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▄@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
э
R
6__inference_stream_2_input_drop_layer_call_fn_13981724

inputs
identity╫
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_139787602
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ▄2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
╛
t
X__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_13982505

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
П№
Т
G__inference_basemodel_layer_call_and_return_conditional_losses_13981416
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
&dense_1_matmul_readvariableop_resource:	└T5
'dense_1_biasadd_readvariableop_resource:TE
7batch_normalization_3_batchnorm_readvariableop_resource:TI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:TG
9batch_normalization_3_batchnorm_readvariableop_1_resource:TG
9batch_normalization_3_batchnorm_readvariableop_2_resource:T
identityИв,batch_normalization/batchnorm/ReadVariableOpв.batch_normalization/batchnorm/ReadVariableOp_1в.batch_normalization/batchnorm/ReadVariableOp_2в0batch_normalization/batchnorm/mul/ReadVariableOpв.batch_normalization_1/batchnorm/ReadVariableOpв0batch_normalization_1/batchnorm/ReadVariableOp_1в0batch_normalization_1/batchnorm/ReadVariableOp_2в2batch_normalization_1/batchnorm/mul/ReadVariableOpв.batch_normalization_2/batchnorm/ReadVariableOpв0batch_normalization_2/batchnorm/ReadVariableOp_1в0batch_normalization_2/batchnorm/ReadVariableOp_2в2batch_normalization_2/batchnorm/mul/ReadVariableOpв.batch_normalization_3/batchnorm/ReadVariableOpв0batch_normalization_3/batchnorm/ReadVariableOp_1в0batch_normalization_3/batchnorm/ReadVariableOp_2в2batch_normalization_3/batchnorm/mul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв&stream_0_conv_1/BiasAdd/ReadVariableOpв2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв&stream_1_conv_1/BiasAdd/ReadVariableOpв2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв&stream_2_conv_1/BiasAdd/ReadVariableOpв2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЙ
stream_2_input_drop/IdentityIdentityinputs_2*
T0*,
_output_shapes
:         ▄2
stream_2_input_drop/IdentityЙ
stream_1_input_drop/IdentityIdentityinputs_1*
T0*,
_output_shapes
:         ▄2
stream_1_input_drop/IdentityЙ
stream_0_input_drop/IdentityIdentityinputs_0*
T0*,
_output_shapes
:         ▄2
stream_0_input_drop/IdentityЩ
%stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2'
%stream_2_conv_1/conv1d/ExpandDims/dimц
!stream_2_conv_1/conv1d/ExpandDims
ExpandDims%stream_2_input_drop/Identity:output:0.stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄2#
!stream_2_conv_1/conv1d/ExpandDimsш
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
'stream_2_conv_1/conv1d/ExpandDims_1/dimў
#stream_2_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_2_conv_1/conv1d/ExpandDims_1ў
stream_2_conv_1/conv1dConv2D*stream_2_conv_1/conv1d/ExpandDims:output:0,stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2
stream_2_conv_1/conv1d├
stream_2_conv_1/conv1d/SqueezeSqueezestream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        2 
stream_2_conv_1/conv1d/Squeeze╝
&stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_2_conv_1/BiasAdd/ReadVariableOp═
stream_2_conv_1/BiasAddBiasAdd'stream_2_conv_1/conv1d/Squeeze:output:0.stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2
stream_2_conv_1/BiasAddЩ
%stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2'
%stream_1_conv_1/conv1d/ExpandDims/dimц
!stream_1_conv_1/conv1d/ExpandDims
ExpandDims%stream_1_input_drop/Identity:output:0.stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄2#
!stream_1_conv_1/conv1d/ExpandDimsш
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
'stream_1_conv_1/conv1d/ExpandDims_1/dimў
#stream_1_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_1_conv_1/conv1d/ExpandDims_1ў
stream_1_conv_1/conv1dConv2D*stream_1_conv_1/conv1d/ExpandDims:output:0,stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2
stream_1_conv_1/conv1d├
stream_1_conv_1/conv1d/SqueezeSqueezestream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        2 
stream_1_conv_1/conv1d/Squeeze╝
&stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_1_conv_1/BiasAdd/ReadVariableOp═
stream_1_conv_1/BiasAddBiasAdd'stream_1_conv_1/conv1d/Squeeze:output:0.stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2
stream_1_conv_1/BiasAddЩ
%stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2'
%stream_0_conv_1/conv1d/ExpandDims/dimц
!stream_0_conv_1/conv1d/ExpandDims
ExpandDims%stream_0_input_drop/Identity:output:0.stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄2#
!stream_0_conv_1/conv1d/ExpandDimsш
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
'stream_0_conv_1/conv1d/ExpandDims_1/dimў
#stream_0_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1ў
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d├
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        2 
stream_0_conv_1/conv1d/Squeeze╝
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp═
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2
stream_0_conv_1/BiasAdd╘
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
%batch_normalization_2/batchnorm/add/yр
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/addе
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp▌
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/mul╫
%batch_normalization_2/batchnorm/mul_1Mul stream_2_conv_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2'
%batch_normalization_2/batchnorm/mul_1┌
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1▌
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/mul_2┌
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2█
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/subт
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2'
%batch_normalization_2/batchnorm/add_1╘
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
%batch_normalization_1/batchnorm/add/yр
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/addе
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_1/batchnorm/Rsqrtр
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp▌
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/mul╫
%batch_normalization_1/batchnorm/mul_1Mul stream_1_conv_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2'
%batch_normalization_1/batchnorm/mul_1┌
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1▌
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_1/batchnorm/mul_2┌
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2█
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/subт
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2'
%batch_normalization_1/batchnorm/add_1╬
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
#batch_normalization/batchnorm/add/y╪
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/addЯ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp╒
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/mul╤
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2%
#batch_normalization/batchnorm/mul_1╘
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1╒
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/mul_2╘
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2╙
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/sub┌
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2%
#batch_normalization/batchnorm/add_1Р
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ▄@2
activation_2/ReluР
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ▄@2
activation_1/ReluК
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ▄@2
activation/ReluШ
stream_2_drop_1/IdentityIdentityactivation_2/Relu:activations:0*
T0*,
_output_shapes
:         ▄@2
stream_2_drop_1/IdentityШ
stream_1_drop_1/IdentityIdentityactivation_1/Relu:activations:0*
T0*,
_output_shapes
:         ▄@2
stream_1_drop_1/IdentityЦ
stream_0_drop_1/IdentityIdentityactivation/Relu:activations:0*
T0*,
_output_shapes
:         ▄@2
stream_0_drop_1/Identityд
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices╒
global_average_pooling1d/MeanMean!stream_0_drop_1/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2
global_average_pooling1d/Meanи
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_1/Mean/reduction_indices█
global_average_pooling1d_1/MeanMean!stream_1_drop_1/Identity:output:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2!
global_average_pooling1d_1/Meanи
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indices█
global_average_pooling1d_2/MeanMean!stream_2_drop_1/Identity:output:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2!
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
:         └2
concatenate/concatР
dense_1_dropout/IdentityIdentityconcatenate/concat:output:0*
T0*(
_output_shapes
:         └2
dense_1_dropout/Identityж
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	└T*
dtype02
dense_1/MatMul/ReadVariableOpж
dense_1/MatMulMatMul!dense_1_dropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
dense_1/BiasAdd╘
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
%batch_normalization_3/batchnorm/add/yр
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/addе
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/Rsqrtр
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOp▌
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
0batch_normalization_3/batchnorm/ReadVariableOp_1▌
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
#batch_normalization_3/batchnorm/sub▌
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_3/batchnorm/add_1а
dense_activation_1/SigmoidSigmoid)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2
dense_activation_1/Sigmoidю
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absй
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulЇ
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareй
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const┌
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulю
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absй
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul╞
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	└T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└T2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const╖
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
╫#<2"
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

Identityм
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_1_conv_1/BiasAdd/ReadVariableOp3^stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_2_conv_1/BiasAdd/ReadVariableOp3^stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         ▄:         ▄:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 2\
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
:         ▄
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:         ▄
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:         ▄
"
_user_specified_name
inputs/2
║+
ь
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_13978270

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
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
moments/StopGradient▒
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                  @2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
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
╫#<2
AssignMovingAvg/decayд
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
╫#<2
AssignMovingAvg_1/decayк
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
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
 :                  @2
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
 :                  @2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                  @2

IdentityЄ
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
к
ш
*__inference_model_1_layer_call_fn_13980394
left_inputs
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

unknown_17:	└T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityИвStatefulPartitionedCallа
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
:         T*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_139802902
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
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:         ▄
%
_user_specified_nameleft_inputs
ў
▓
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_13978606

inputs/
!batchnorm_readvariableop_resource:T3
%batchnorm_mul_readvariableop_resource:T1
#batchnorm_readvariableop_1_resource:T1
#batchnorm_readvariableop_2_resource:T
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
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
:         T2
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
╬*
ь
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_13982656

inputs5
'assignmovingavg_readvariableop_resource:T7
)assignmovingavg_1_readvariableop_resource:T3
%batchnorm_mul_readvariableop_resource:T/
!batchnorm_readvariableop_resource:T
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpК
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
moments/StopGradientд
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         T2
moments/SquaredDifferenceТ
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
╫#<2
AssignMovingAvg/decayд
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
╫#<2
AssignMovingAvg_1/decayк
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:T*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg_1/subЧ
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
:         T2
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
:         T2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         T2

IdentityЄ
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
ъ
l
P__inference_dense_activation_1_layer_call_and_return_conditional_losses_13982666

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
хб
ш
G__inference_basemodel_layer_call_and_return_conditional_losses_13979718

inputs
inputs_1
inputs_2.
stream_2_conv_1_13979625:@&
stream_2_conv_1_13979627:@.
stream_1_conv_1_13979630:@&
stream_1_conv_1_13979632:@.
stream_0_conv_1_13979635:@&
stream_0_conv_1_13979637:@,
batch_normalization_2_13979640:@,
batch_normalization_2_13979642:@,
batch_normalization_2_13979644:@,
batch_normalization_2_13979646:@,
batch_normalization_1_13979649:@,
batch_normalization_1_13979651:@,
batch_normalization_1_13979653:@,
batch_normalization_1_13979655:@*
batch_normalization_13979658:@*
batch_normalization_13979660:@*
batch_normalization_13979662:@*
batch_normalization_13979664:@#
dense_1_13979678:	└T
dense_1_13979680:T,
batch_normalization_3_13979683:T,
batch_normalization_3_13979685:T,
batch_normalization_3_13979687:T,
batch_normalization_3_13979689:T
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв'stream_0_conv_1/StatefulPartitionedCallв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв'stream_0_drop_1/StatefulPartitionedCallв+stream_0_input_drop/StatefulPartitionedCallв'stream_1_conv_1/StatefulPartitionedCallв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв'stream_1_drop_1/StatefulPartitionedCallв+stream_1_input_drop/StatefulPartitionedCallв'stream_2_conv_1/StatefulPartitionedCallв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpв'stream_2_drop_1/StatefulPartitionedCallв+stream_2_input_drop/StatefulPartitionedCallЩ
+stream_2_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_139795552-
+stream_2_input_drop/StatefulPartitionedCall╟
+stream_1_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_1,^stream_2_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_139795322-
+stream_1_input_drop/StatefulPartitionedCall┼
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs,^stream_1_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_139795092-
+stream_0_input_drop/StatefulPartitionedCallє
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_2_input_drop/StatefulPartitionedCall:output:0stream_2_conv_1_13979625stream_2_conv_1_13979627*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_139787972)
'stream_2_conv_1/StatefulPartitionedCallє
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_1_input_drop/StatefulPartitionedCall:output:0stream_1_conv_1_13979630stream_1_conv_1_13979632*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_139788242)
'stream_1_conv_1/StatefulPartitionedCallє
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_13979635stream_0_conv_1_13979637*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_139788512)
'stream_0_conv_1/StatefulPartitionedCall╧
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_13979640batch_normalization_2_13979642batch_normalization_2_13979644batch_normalization_2_13979646*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_139794482/
-batch_normalization_2/StatefulPartitionedCall╧
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_13979649batch_normalization_1_13979651batch_normalization_1_13979653batch_normalization_1_13979655*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_139793882/
-batch_normalization_1/StatefulPartitionedCall┴
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_13979658batch_normalization_13979660batch_normalization_13979662batch_normalization_13979664*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_139793282-
+batch_normalization/StatefulPartitionedCallЪ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_activation_2_layer_call_and_return_conditional_losses_139789492
activation_2/PartitionedCallЪ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_activation_1_layer_call_and_return_conditional_losses_139789562
activation_1/PartitionedCallТ
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_139789632
activation/PartitionedCall╪
'stream_2_drop_1/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_139792582)
'stream_2_drop_1/StatefulPartitionedCall╘
'stream_1_drop_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0(^stream_2_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_139792352)
'stream_1_drop_1/StatefulPartitionedCall╥
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0(^stream_1_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_139792122)
'stream_0_drop_1/StatefulPartitionedCall│
(global_average_pooling1d/PartitionedCallPartitionedCall0stream_0_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_139789912*
(global_average_pooling1d/PartitionedCall╣
*global_average_pooling1d_1/PartitionedCallPartitionedCall0stream_1_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *a
f\RZ
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_139789982,
*global_average_pooling1d_1/PartitionedCall╣
*global_average_pooling1d_2/PartitionedCallPartitionedCall0stream_2_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *a
f\RZ
X__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_139790052,
*global_average_pooling1d_2/PartitionedCall·
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:03global_average_pooling1d_1/PartitionedCall:output:03global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_139790152
concatenate/PartitionedCallН
dense_1_dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_dense_1_dropout_layer_call_and_return_conditional_losses_139791662!
dense_1_dropout/PartitionedCall║
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_13979678dense_1_13979680*
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
GPU2*0J 8В *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_139790402!
dense_1/StatefulPartitionedCall┬
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_13979683batch_normalization_3_13979685batch_normalization_3_13979687batch_normalization_3_13979689*
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
GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_139786662/
-batch_normalization_3/StatefulPartitionedCallз
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
GPU2*0J 8В *Y
fTRR
P__inference_dense_activation_1_layer_call_and_return_conditional_losses_139790602$
"dense_activation_1/PartitionedCall╦
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_13979635*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absй
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul╤
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_13979630*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareй
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const┌
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul╦
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_13979625*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absй
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul░
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_13979678*
_output_shapes
:	└T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└T2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const╖
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЖ
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

IdentityП
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall(^stream_1_conv_1/StatefulPartitionedCall9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_1_drop_1/StatefulPartitionedCall,^stream_1_input_drop/StatefulPartitionedCall(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_2_drop_1/StatefulPartitionedCall,^stream_2_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         ▄:         ▄:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:         ▄
 
_user_specified_nameinputs:TP
,
_output_shapes
:         ▄
 
_user_specified_nameinputs:TP
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
О
░
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_13978934

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2
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
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▄@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
у
╘
M__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_13981782

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄2
conv1d/ExpandDims╕
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
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2	
BiasAdd▐
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absй
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identity─
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▄: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
╖
░
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_13978048

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
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
 :                  @2
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
О
k
M__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_13982379

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ▄@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ▄@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
Ї
d
H__inference_activation_layer_call_and_return_conditional_losses_13978963

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         ▄@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         ▄@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
ў
l
M__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_13979258

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ▄@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ▄@*
dtype0*
seed╖*
seed2╖2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ▄@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
║+
ь
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13978432

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
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
moments/StopGradient▒
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                  @2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
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
╫#<2
AssignMovingAvg/decayд
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
╫#<2
AssignMovingAvg_1/decayк
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
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
 :                  @2
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
 :                  @2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                  @2

IdentityЄ
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
Я
г
2__inference_stream_1_conv_1_layer_call_fn_13981797

inputs
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_139788242
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▄: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
ў
l
M__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_13979235

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ▄@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ▄@*
dtype0*
seed╖*
seed2╖2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ▄@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
Ы
у
*__inference_model_1_layer_call_fn_13980733

inputs
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

unknown_17:	└T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityИвStatefulPartitionedCallЫ
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
	
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_139802902
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
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
С	
╙
8__inference_batch_normalization_1_layer_call_fn_13982040

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallл
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
GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_139782702
StatefulPartitionedCallИ
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
Я
г
2__inference_stream_2_conv_1_layer_call_fn_13981833

inputs
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_139787972
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▄: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
З+
ъ
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_13979328

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
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
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         ▄@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
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
╫#<2
AssignMovingAvg/decayд
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
╫#<2
AssignMovingAvg_1/decayк
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2
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
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ▄@2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▄@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
╖
░
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_13981926

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
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
 :                  @2
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
Й+
ь
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13982334

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
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
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         ▄@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
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
╫#<2
AssignMovingAvg/decayд
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
╫#<2
AssignMovingAvg_1/decayк
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2
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
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ▄@2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▄@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
Ў
f
J__inference_activation_1_layer_call_and_return_conditional_losses_13982354

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         ▄@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         ▄@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
╛
t
X__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_13978568

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
у
╘
M__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_13978797

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄2
conv1d/ExpandDims╕
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
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2	
BiasAdd▐
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absй
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identity─
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▄: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
у
╘
M__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_13978851

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄2
conv1d/ExpandDims╕
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
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2	
BiasAdd▐
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absй
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identity─
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▄: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
▀
Б
I__inference_concatenate_layer_call_and_return_conditional_losses_13979015

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
:         └2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         @:         @:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_nameinputs
╨
o
6__inference_stream_0_input_drop_layer_call_fn_13981675

inputs
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_139795092
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ▄2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
М
i
M__inference_dense_1_dropout_layer_call_and_return_conditional_losses_13982545

inputs
identity[
IdentityIdentityinputs*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         └:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
О
k
M__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_13982433

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ▄@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ▄@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
К
t
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_13982489

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
:         @2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
ё
Y
=__inference_global_average_pooling1d_2_layer_call_fn_13982499

inputs
identity┘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *a
f\RZ
X__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_139790052
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
╨
o
6__inference_stream_1_input_drop_layer_call_fn_13981702

inputs
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_139795322
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ▄2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
╫
Q
5__inference_dense_activation_1_layer_call_fn_13982661

inputs
identity╤
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
GPU2*0J 8В *Y
fTRR
P__inference_dense_activation_1_layer_call_and_return_conditional_losses_139790602
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
╬*
ь
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_13978666

inputs5
'assignmovingavg_readvariableop_resource:T7
)assignmovingavg_1_readvariableop_resource:T3
%batchnorm_mul_readvariableop_resource:T/
!batchnorm_readvariableop_resource:T
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpК
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
moments/StopGradientд
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         T2
moments/SquaredDifferenceТ
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
╫#<2
AssignMovingAvg/decayд
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
╫#<2
AssignMovingAvg_1/decayк
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:T*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2
AssignMovingAvg_1/subЧ
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
:         T2
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
:         T2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         T2

IdentityЄ
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
М
ф
&__inference_signature_wrapper_13980627
left_inputs
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

unknown_17:	└T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityИвStatefulPartitionedCallЖ
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
:         T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference__wrapped_model_139780242
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
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:         ▄
%
_user_specified_nameleft_inputs
▄
╙
8__inference_batch_normalization_3_layer_call_fn_13982602

inputs
unknown:T
	unknown_0:T
	unknown_1:T
	unknown_2:T
identityИвStatefulPartitionedCallЮ
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
GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_139786662
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
К
t
X__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_13982511

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
:         @2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
О
k
M__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_13982406

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ▄@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ▄@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
х
N
2__inference_stream_0_drop_1_layer_call_fn_13982369

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_139789842
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ▄@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
К
t
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_13978998

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
:         @2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
∙
Ш
*__inference_dense_1_layer_call_fn_13982560

inputs
unknown:	└T
	unknown_0:T
identityИвStatefulPartitionedCall°
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
GPU2*0J 8В *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_139790402
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
:         └: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
╒
N
2__inference_dense_1_dropout_layer_call_fn_13982536

inputs
identity╧
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_dense_1_dropout_layer_call_and_return_conditional_losses_139791662
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         └:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
у
╘
M__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_13981854

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄2
conv1d/ExpandDims╕
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
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2	
BiasAdd▐
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absй
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identity─
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▄: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
Т
o
Q__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_13978760

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ▄2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ▄2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
У	
╙
8__inference_batch_normalization_1_layer_call_fn_13982027

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallн
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
GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_139782102
StatefulPartitionedCallИ
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
√
p
Q__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_13981719

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ▄2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ▄*
dtype0*
seed╖*
seed2╖2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ▄2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
У	
╙
8__inference_batch_normalization_2_layer_call_fn_13982187

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallн
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
GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_139783722
StatefulPartitionedCallИ
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
є
╙
8__inference_batch_normalization_1_layer_call_fn_13982053

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_139789052
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▄@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
√
p
Q__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_13981746

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ▄2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ▄*
dtype0*
seed╖*
seed2╖2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ▄2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
я
╤
6__inference_batch_normalization_layer_call_fn_13981893

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_139789342
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▄@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
Р
▓
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13978876

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2
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
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▄@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
Ё=
╡	
E__inference_model_1_layer_call_and_return_conditional_losses_13980548
left_inputs(
basemodel_13980474:@ 
basemodel_13980476:@(
basemodel_13980478:@ 
basemodel_13980480:@(
basemodel_13980482:@ 
basemodel_13980484:@ 
basemodel_13980486:@ 
basemodel_13980488:@ 
basemodel_13980490:@ 
basemodel_13980492:@ 
basemodel_13980494:@ 
basemodel_13980496:@ 
basemodel_13980498:@ 
basemodel_13980500:@ 
basemodel_13980502:@ 
basemodel_13980504:@ 
basemodel_13980506:@ 
basemodel_13980508:@%
basemodel_13980510:	└T 
basemodel_13980512:T 
basemodel_13980514:T 
basemodel_13980516:T 
basemodel_13980518:T 
basemodel_13980520:T
identityИв!basemodel/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЯ
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsleft_inputsleft_inputsbasemodel_13980474basemodel_13980476basemodel_13980478basemodel_13980480basemodel_13980482basemodel_13980484basemodel_13980486basemodel_13980488basemodel_13980490basemodel_13980492basemodel_13980494basemodel_13980496basemodel_13980498basemodel_13980500basemodel_13980502basemodel_13980504basemodel_13980506basemodel_13980508basemodel_13980510basemodel_13980512basemodel_13980514basemodel_13980516basemodel_13980518basemodel_13980520*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_basemodel_layer_call_and_return_conditional_losses_139797182#
!basemodel/StatefulPartitionedCall┼
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_13980482*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absй
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul╦
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_13980478*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareй
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const┌
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┼
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_13980474*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absй
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul▓
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_13980510*
_output_shapes
:	└T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└T2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const╖
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЕ
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

Identity═
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:Y U
,
_output_shapes
:         ▄
%
_user_specified_nameleft_inputs
└
Г
,__inference_basemodel_layer_call_fn_13979138
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

unknown_17:	└T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityИвStatefulPartitionedCall╜
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
:         T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_basemodel_layer_call_and_return_conditional_losses_139790872
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
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         ▄:         ▄:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:         ▄
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:         ▄
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:         ▄
"
_user_specified_name
inputs_2
▓
ш
*__inference_model_1_layer_call_fn_13980158
left_inputs
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

unknown_17:	└T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityИвStatefulPartitionedCallи
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
:         T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_139801072
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
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:         ▄
%
_user_specified_nameleft_inputs
╣
▓
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13978372

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
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
 :                  @2
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
█
I
-__inference_activation_layer_call_fn_13982339

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_139789632
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ▄@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
√
p
Q__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_13979532

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ▄2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ▄*
dtype0*
seed╖*
seed2╖2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ▄2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
э
╤
6__inference_batch_normalization_layer_call_fn_13981906

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_139793282
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▄@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
╚
k
2__inference_stream_2_drop_1_layer_call_fn_13982428

inputs
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_139792582
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
ў
l
M__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_13982445

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ▄@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ▄@*
dtype0*
seed╖*
seed2╖2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ▄@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
щ
Г
I__inference_concatenate_layer_call_and_return_conditional_losses_13982526
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
:         └2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         @:         @:         @:Q M
'
_output_shapes
:         @
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         @
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:         @
"
_user_specified_name
inputs/2
╠
┐
__inference_loss_fn_2_13982699T
>stream_2_conv_1_kernel_regularizer_abs_readvariableop_resource:@
identityИв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpё
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_2_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absй
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
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
О
k
M__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_13978970

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ▄@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ▄@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
▀
K
/__inference_activation_2_layer_call_fn_13982359

inputs
identity╨
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_activation_2_layer_call_and_return_conditional_losses_139789492
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ▄@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
П	
╤
6__inference_batch_normalization_layer_call_fn_13981867

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallл
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
GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_139780482
StatefulPartitionedCallИ
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
г
у
*__inference_model_1_layer_call_fn_13980680

inputs
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

unknown_17:	└T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityИвStatefulPartitionedCallг
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
GPU2*0J 8В *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_139801072
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
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
╛
t
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_13978544

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
Т
o
Q__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_13981680

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ▄2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ▄2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
╕
Г
,__inference_basemodel_layer_call_fn_13979824
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

unknown_17:	└T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityИвStatefulPartitionedCall╡
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
:         T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_basemodel_layer_call_and_return_conditional_losses_139797182
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
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         ▄:         ▄:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:         ▄
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:         ▄
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:         ▄
"
_user_specified_name
inputs_2
у║
ж
G__inference_basemodel_layer_call_and_return_conditional_losses_13981665
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
&dense_1_matmul_readvariableop_resource:	└T5
'dense_1_biasadd_readvariableop_resource:TK
=batch_normalization_3_assignmovingavg_readvariableop_resource:TM
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:TI
;batch_normalization_3_batchnorm_mul_readvariableop_resource:TE
7batch_normalization_3_batchnorm_readvariableop_resource:T
identityИв#batch_normalization/AssignMovingAvgв2batch_normalization/AssignMovingAvg/ReadVariableOpв%batch_normalization/AssignMovingAvg_1в4batch_normalization/AssignMovingAvg_1/ReadVariableOpв,batch_normalization/batchnorm/ReadVariableOpв0batch_normalization/batchnorm/mul/ReadVariableOpв%batch_normalization_1/AssignMovingAvgв4batch_normalization_1/AssignMovingAvg/ReadVariableOpв'batch_normalization_1/AssignMovingAvg_1в6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpв.batch_normalization_1/batchnorm/ReadVariableOpв2batch_normalization_1/batchnorm/mul/ReadVariableOpв%batch_normalization_2/AssignMovingAvgв4batch_normalization_2/AssignMovingAvg/ReadVariableOpв'batch_normalization_2/AssignMovingAvg_1в6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpв.batch_normalization_2/batchnorm/ReadVariableOpв2batch_normalization_2/batchnorm/mul/ReadVariableOpв%batch_normalization_3/AssignMovingAvgв4batch_normalization_3/AssignMovingAvg/ReadVariableOpв'batch_normalization_3/AssignMovingAvg_1в6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpв.batch_normalization_3/batchnorm/ReadVariableOpв2batch_normalization_3/batchnorm/mul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв&stream_0_conv_1/BiasAdd/ReadVariableOpв2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв&stream_1_conv_1/BiasAdd/ReadVariableOpв2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв&stream_2_conv_1/BiasAdd/ReadVariableOpв2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЛ
!stream_2_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2#
!stream_2_input_drop/dropout/Const╢
stream_2_input_drop/dropout/MulMulinputs_2*stream_2_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:         ▄2!
stream_2_input_drop/dropout/Mul~
!stream_2_input_drop/dropout/ShapeShapeinputs_2*
T0*
_output_shapes
:2#
!stream_2_input_drop/dropout/ShapeР
8stream_2_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_2_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:         ▄*
dtype0*
seed╖*
seed2╖2:
8stream_2_input_drop/dropout/random_uniform/RandomUniformЭ
*stream_2_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2,
*stream_2_input_drop/dropout/GreaterEqual/yУ
(stream_2_input_drop/dropout/GreaterEqualGreaterEqualAstream_2_input_drop/dropout/random_uniform/RandomUniform:output:03stream_2_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄2*
(stream_2_input_drop/dropout/GreaterEqual└
 stream_2_input_drop/dropout/CastCast,stream_2_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄2"
 stream_2_input_drop/dropout/Cast╧
!stream_2_input_drop/dropout/Mul_1Mul#stream_2_input_drop/dropout/Mul:z:0$stream_2_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄2#
!stream_2_input_drop/dropout/Mul_1Л
!stream_1_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2#
!stream_1_input_drop/dropout/Const╢
stream_1_input_drop/dropout/MulMulinputs_1*stream_1_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:         ▄2!
stream_1_input_drop/dropout/Mul~
!stream_1_input_drop/dropout/ShapeShapeinputs_1*
T0*
_output_shapes
:2#
!stream_1_input_drop/dropout/ShapeР
8stream_1_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_1_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:         ▄*
dtype0*
seed╖*
seed2╖2:
8stream_1_input_drop/dropout/random_uniform/RandomUniformЭ
*stream_1_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2,
*stream_1_input_drop/dropout/GreaterEqual/yУ
(stream_1_input_drop/dropout/GreaterEqualGreaterEqualAstream_1_input_drop/dropout/random_uniform/RandomUniform:output:03stream_1_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄2*
(stream_1_input_drop/dropout/GreaterEqual└
 stream_1_input_drop/dropout/CastCast,stream_1_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄2"
 stream_1_input_drop/dropout/Cast╧
!stream_1_input_drop/dropout/Mul_1Mul#stream_1_input_drop/dropout/Mul:z:0$stream_1_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄2#
!stream_1_input_drop/dropout/Mul_1Л
!stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2#
!stream_0_input_drop/dropout/Const╢
stream_0_input_drop/dropout/MulMulinputs_0*stream_0_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:         ▄2!
stream_0_input_drop/dropout/Mul~
!stream_0_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/ShapeР
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:         ▄*
dtype0*
seed╖*
seed2╖2:
8stream_0_input_drop/dropout/random_uniform/RandomUniformЭ
*stream_0_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2,
*stream_0_input_drop/dropout/GreaterEqual/yУ
(stream_0_input_drop/dropout/GreaterEqualGreaterEqualAstream_0_input_drop/dropout/random_uniform/RandomUniform:output:03stream_0_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄2*
(stream_0_input_drop/dropout/GreaterEqual└
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄2"
 stream_0_input_drop/dropout/Cast╧
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄2#
!stream_0_input_drop/dropout/Mul_1Щ
%stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2'
%stream_2_conv_1/conv1d/ExpandDims/dimц
!stream_2_conv_1/conv1d/ExpandDims
ExpandDims%stream_2_input_drop/dropout/Mul_1:z:0.stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄2#
!stream_2_conv_1/conv1d/ExpandDimsш
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
'stream_2_conv_1/conv1d/ExpandDims_1/dimў
#stream_2_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_2_conv_1/conv1d/ExpandDims_1ў
stream_2_conv_1/conv1dConv2D*stream_2_conv_1/conv1d/ExpandDims:output:0,stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2
stream_2_conv_1/conv1d├
stream_2_conv_1/conv1d/SqueezeSqueezestream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        2 
stream_2_conv_1/conv1d/Squeeze╝
&stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_2_conv_1/BiasAdd/ReadVariableOp═
stream_2_conv_1/BiasAddBiasAdd'stream_2_conv_1/conv1d/Squeeze:output:0.stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2
stream_2_conv_1/BiasAddЩ
%stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2'
%stream_1_conv_1/conv1d/ExpandDims/dimц
!stream_1_conv_1/conv1d/ExpandDims
ExpandDims%stream_1_input_drop/dropout/Mul_1:z:0.stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄2#
!stream_1_conv_1/conv1d/ExpandDimsш
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
'stream_1_conv_1/conv1d/ExpandDims_1/dimў
#stream_1_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_1_conv_1/conv1d/ExpandDims_1ў
stream_1_conv_1/conv1dConv2D*stream_1_conv_1/conv1d/ExpandDims:output:0,stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2
stream_1_conv_1/conv1d├
stream_1_conv_1/conv1d/SqueezeSqueezestream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        2 
stream_1_conv_1/conv1d/Squeeze╝
&stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_1_conv_1/BiasAdd/ReadVariableOp═
stream_1_conv_1/BiasAddBiasAdd'stream_1_conv_1/conv1d/Squeeze:output:0.stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2
stream_1_conv_1/BiasAddЩ
%stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2'
%stream_0_conv_1/conv1d/ExpandDims/dimц
!stream_0_conv_1/conv1d/ExpandDims
ExpandDims%stream_0_input_drop/dropout/Mul_1:z:0.stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄2#
!stream_0_conv_1/conv1d/ExpandDimsш
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
'stream_0_conv_1/conv1d/ExpandDims_1/dimў
#stream_0_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1ў
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d├
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        2 
stream_0_conv_1/conv1d/Squeeze╝
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp═
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2
stream_0_conv_1/BiasAdd╜
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_2/moments/mean/reduction_indicesя
"batch_normalization_2/moments/meanMean stream_2_conv_1/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2$
"batch_normalization_2/moments/mean┬
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*"
_output_shapes
:@2,
*batch_normalization_2/moments/StopGradientЕ
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference stream_2_conv_1/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:         ▄@21
/batch_normalization_2/moments/SquaredDifference┼
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
&batch_normalization_2/moments/variance├
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze╦
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
╫#<2-
+batch_normalization_2/AssignMovingAvg/decayц
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpЁ
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/subч
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/mulн
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_2/AssignMovingAvgг
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_2/AssignMovingAvg_1/decayь
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp°
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/subя
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/mul╖
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
%batch_normalization_2/batchnorm/add/y┌
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/addе
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp▌
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/mul╫
%batch_normalization_2/batchnorm/mul_1Mul stream_2_conv_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2'
%batch_normalization_2/batchnorm/mul_1╙
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/mul_2╘
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp┘
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/subт
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2'
%batch_normalization_2/batchnorm/add_1╜
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indicesя
"batch_normalization_1/moments/meanMean stream_1_conv_1/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2$
"batch_normalization_1/moments/mean┬
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:@2,
*batch_normalization_1/moments/StopGradientЕ
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_1_conv_1/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:         ▄@21
/batch_normalization_1/moments/SquaredDifference┼
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
&batch_normalization_1/moments/variance├
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze╦
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
╫#<2-
+batch_normalization_1/AssignMovingAvg/decayц
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOpЁ
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_1/AssignMovingAvg/subч
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_1/AssignMovingAvg/mulн
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_1/AssignMovingAvgг
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_1/AssignMovingAvg_1/decayь
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp°
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_1/AssignMovingAvg_1/subя
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_1/AssignMovingAvg_1/mul╖
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
%batch_normalization_1/batchnorm/add/y┌
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/addе
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_1/batchnorm/Rsqrtр
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp▌
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/mul╫
%batch_normalization_1/batchnorm/mul_1Mul stream_1_conv_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2'
%batch_normalization_1/batchnorm/mul_1╙
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_1/batchnorm/mul_2╘
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp┘
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/subт
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2'
%batch_normalization_1/batchnorm/add_1╣
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indicesщ
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
(batch_normalization/moments/StopGradient 
-batch_normalization/moments/SquaredDifferenceSquaredDifference stream_0_conv_1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:         ▄@2/
-batch_normalization/moments/SquaredDifference┴
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
$batch_normalization/moments/variance╜
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
%batch_normalization/moments/Squeeze_1Ы
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2+
)batch_normalization/AssignMovingAvg/decayр
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpш
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:@2)
'batch_normalization/AssignMovingAvg/sub▀
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2)
'batch_normalization/AssignMovingAvg/mulг
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
╫#<2-
+batch_normalization/AssignMovingAvg_1/decayц
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOpЁ
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2+
)batch_normalization/AssignMovingAvg_1/subч
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2+
)batch_normalization/AssignMovingAvg_1/mulн
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
#batch_normalization/batchnorm/add/y╥
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/addЯ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp╒
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/mul╤
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2%
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
,batch_normalization/batchnorm/ReadVariableOp╤
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/sub┌
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2%
#batch_normalization/batchnorm/add_1Р
activation_2/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ▄@2
activation_2/ReluР
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ▄@2
activation_1/ReluК
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ▄@2
activation/ReluГ
stream_2_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
stream_2_drop_1/dropout/Const┴
stream_2_drop_1/dropout/MulMulactivation_2/Relu:activations:0&stream_2_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:         ▄@2
stream_2_drop_1/dropout/MulН
stream_2_drop_1/dropout/ShapeShapeactivation_2/Relu:activations:0*
T0*
_output_shapes
:2
stream_2_drop_1/dropout/ShapeД
4stream_2_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_2_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:         ▄@*
dtype0*
seed╖*
seed2╖26
4stream_2_drop_1/dropout/random_uniform/RandomUniformХ
&stream_2_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2(
&stream_2_drop_1/dropout/GreaterEqual/yГ
$stream_2_drop_1/dropout/GreaterEqualGreaterEqual=stream_2_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_2_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄@2&
$stream_2_drop_1/dropout/GreaterEqual┤
stream_2_drop_1/dropout/CastCast(stream_2_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄@2
stream_2_drop_1/dropout/Cast┐
stream_2_drop_1/dropout/Mul_1Mulstream_2_drop_1/dropout/Mul:z:0 stream_2_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄@2
stream_2_drop_1/dropout/Mul_1Г
stream_1_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
stream_1_drop_1/dropout/Const┴
stream_1_drop_1/dropout/MulMulactivation_1/Relu:activations:0&stream_1_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:         ▄@2
stream_1_drop_1/dropout/MulН
stream_1_drop_1/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
stream_1_drop_1/dropout/ShapeД
4stream_1_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_1_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:         ▄@*
dtype0*
seed╖*
seed2╖26
4stream_1_drop_1/dropout/random_uniform/RandomUniformХ
&stream_1_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2(
&stream_1_drop_1/dropout/GreaterEqual/yГ
$stream_1_drop_1/dropout/GreaterEqualGreaterEqual=stream_1_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_1_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄@2&
$stream_1_drop_1/dropout/GreaterEqual┤
stream_1_drop_1/dropout/CastCast(stream_1_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄@2
stream_1_drop_1/dropout/Cast┐
stream_1_drop_1/dropout/Mul_1Mulstream_1_drop_1/dropout/Mul:z:0 stream_1_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄@2
stream_1_drop_1/dropout/Mul_1Г
stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
stream_0_drop_1/dropout/Const┐
stream_0_drop_1/dropout/MulMulactivation/Relu:activations:0&stream_0_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:         ▄@2
stream_0_drop_1/dropout/MulЛ
stream_0_drop_1/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/ShapeД
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:         ▄@*
dtype0*
seed╖*
seed2╖26
4stream_0_drop_1/dropout/random_uniform/RandomUniformХ
&stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2(
&stream_0_drop_1/dropout/GreaterEqual/yГ
$stream_0_drop_1/dropout/GreaterEqualGreaterEqual=stream_0_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄@2&
$stream_0_drop_1/dropout/GreaterEqual┤
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄@2
stream_0_drop_1/dropout/Cast┐
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄@2
stream_0_drop_1/dropout/Mul_1д
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices╒
global_average_pooling1d/MeanMean!stream_0_drop_1/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2
global_average_pooling1d/Meanи
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_1/Mean/reduction_indices█
global_average_pooling1d_1/MeanMean!stream_1_drop_1/dropout/Mul_1:z:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2!
global_average_pooling1d_1/Meanи
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indices█
global_average_pooling1d_2/MeanMean!stream_2_drop_1/dropout/Mul_1:z:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2!
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
:         └2
concatenate/concatж
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	└T*
dtype02
dense_1/MatMul/ReadVariableOpа
dense_1/MatMulMatMulconcatenate/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
dense_1/BiasAdd╢
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indicesу
"batch_normalization_3/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2$
"batch_normalization_3/moments/mean╛
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:T2,
*batch_normalization_3/moments/StopGradient°
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:         T21
/batch_normalization_3/moments/SquaredDifference╛
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
'batch_normalization_3/moments/Squeeze_1Я
+batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_3/AssignMovingAvg/decayц
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
:T*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOpЁ
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_3/AssignMovingAvg/subч
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_3/AssignMovingAvg/mulн
%batch_normalization_3/AssignMovingAvgAssignSubVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_3/AssignMovingAvgг
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_3/AssignMovingAvg_1/decayь
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:T*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp°
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_3/AssignMovingAvg_1/subя
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_3/AssignMovingAvg_1/mul╖
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
%batch_normalization_3/batchnorm/add/y┌
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/addе
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/Rsqrtр
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOp▌
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/mul╩
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_3/batchnorm/mul_1╙
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_3/batchnorm/mul_2╘
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp┘
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_3/batchnorm/sub▌
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_3/batchnorm/add_1а
dense_activation_1/SigmoidSigmoid)batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2
dense_activation_1/Sigmoidю
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absй
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulЇ
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareй
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const┌
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulю
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absй
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul╞
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	└T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└T2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const╖
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
╫#<2"
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

IdentityШ
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_1_conv_1/BiasAdd/ReadVariableOp3^stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_2_conv_1/BiasAdd/ReadVariableOp3^stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         ▄:         ▄:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 2J
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
:         ▄
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:         ▄
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:         ▄
"
_user_specified_name
inputs/2
║+
ь
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_13982120

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
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
moments/StopGradient▒
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                  @2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
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
╫#<2
AssignMovingAvg/decayд
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
╫#<2
AssignMovingAvg_1/decayк
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
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
 :                  @2
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
 :                  @2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                  @2

IdentityЄ
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
э
R
6__inference_stream_1_input_drop_layer_call_fn_13981697

inputs
identity╫
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_139787672
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ▄2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
Р
▓
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_13978905

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2
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
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▄@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
ё
╙
8__inference_batch_normalization_2_layer_call_fn_13982226

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_139794482
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▄@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
╕+
ъ
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_13981960

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
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
moments/StopGradient▒
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                  @2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
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
╫#<2
AssignMovingAvg/decayд
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
╫#<2
AssignMovingAvg_1/decayк
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
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
 :                  @2
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
 :                  @2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                  @2

IdentityЄ
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
Ї
d
H__inference_activation_layer_call_and_return_conditional_losses_13982344

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         ▄@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         ▄@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
√
p
Q__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_13979555

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ▄2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ▄*
dtype0*
seed╖*
seed2╖2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ▄2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
х
N
2__inference_stream_2_drop_1_layer_call_fn_13982423

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_139789702
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ▄@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
ў
▓
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_13982622

inputs/
!batchnorm_readvariableop_resource:T3
%batchnorm_mul_readvariableop_resource:T1
#batchnorm_readvariableop_1_resource:T1
#batchnorm_readvariableop_2_resource:T
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
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
:         T2
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
пЦ
р
G__inference_basemodel_layer_call_and_return_conditional_losses_13979087

inputs
inputs_1
inputs_2.
stream_2_conv_1_13978798:@&
stream_2_conv_1_13978800:@.
stream_1_conv_1_13978825:@&
stream_1_conv_1_13978827:@.
stream_0_conv_1_13978852:@&
stream_0_conv_1_13978854:@,
batch_normalization_2_13978877:@,
batch_normalization_2_13978879:@,
batch_normalization_2_13978881:@,
batch_normalization_2_13978883:@,
batch_normalization_1_13978906:@,
batch_normalization_1_13978908:@,
batch_normalization_1_13978910:@,
batch_normalization_1_13978912:@*
batch_normalization_13978935:@*
batch_normalization_13978937:@*
batch_normalization_13978939:@*
batch_normalization_13978941:@#
dense_1_13979041:	└T
dense_1_13979043:T,
batch_normalization_3_13979046:T,
batch_normalization_3_13979048:T,
batch_normalization_3_13979050:T,
batch_normalization_3_13979052:T
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв'stream_0_conv_1/StatefulPartitionedCallв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв'stream_1_conv_1/StatefulPartitionedCallв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв'stream_2_conv_1/StatefulPartitionedCallв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpБ
#stream_2_input_drop/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_139787602%
#stream_2_input_drop/PartitionedCallБ
#stream_1_input_drop/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_139787672%
#stream_1_input_drop/PartitionedCall 
#stream_0_input_drop/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_139787742%
#stream_0_input_drop/PartitionedCallы
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_2_input_drop/PartitionedCall:output:0stream_2_conv_1_13978798stream_2_conv_1_13978800*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_139787972)
'stream_2_conv_1/StatefulPartitionedCallы
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_1_input_drop/PartitionedCall:output:0stream_1_conv_1_13978825stream_1_conv_1_13978827*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_139788242)
'stream_1_conv_1/StatefulPartitionedCallы
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_13978852stream_0_conv_1_13978854*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_139788512)
'stream_0_conv_1/StatefulPartitionedCall╤
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_13978877batch_normalization_2_13978879batch_normalization_2_13978881batch_normalization_2_13978883*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_139788762/
-batch_normalization_2/StatefulPartitionedCall╤
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_13978906batch_normalization_1_13978908batch_normalization_1_13978910batch_normalization_1_13978912*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_139789052/
-batch_normalization_1/StatefulPartitionedCall├
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_13978935batch_normalization_13978937batch_normalization_13978939batch_normalization_13978941*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_139789342-
+batch_normalization/StatefulPartitionedCallЪ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_activation_2_layer_call_and_return_conditional_losses_139789492
activation_2/PartitionedCallЪ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_activation_1_layer_call_and_return_conditional_losses_139789562
activation_1/PartitionedCallТ
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_139789632
activation/PartitionedCallТ
stream_2_drop_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_139789702!
stream_2_drop_1/PartitionedCallТ
stream_1_drop_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_139789772!
stream_1_drop_1/PartitionedCallР
stream_0_drop_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_139789842!
stream_0_drop_1/PartitionedCallл
(global_average_pooling1d/PartitionedCallPartitionedCall(stream_0_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_139789912*
(global_average_pooling1d/PartitionedCall▒
*global_average_pooling1d_1/PartitionedCallPartitionedCall(stream_1_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *a
f\RZ
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_139789982,
*global_average_pooling1d_1/PartitionedCall▒
*global_average_pooling1d_2/PartitionedCallPartitionedCall(stream_2_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *a
f\RZ
X__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_139790052,
*global_average_pooling1d_2/PartitionedCall·
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:03global_average_pooling1d_1/PartitionedCall:output:03global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_139790152
concatenate/PartitionedCallН
dense_1_dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_dense_1_dropout_layer_call_and_return_conditional_losses_139790222!
dense_1_dropout/PartitionedCall║
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_13979041dense_1_13979043*
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
GPU2*0J 8В *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_139790402!
dense_1/StatefulPartitionedCall─
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_13979046batch_normalization_3_13979048batch_normalization_3_13979050batch_normalization_3_13979052*
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
GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_139786062/
-batch_normalization_3/StatefulPartitionedCallз
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
GPU2*0J 8В *Y
fTRR
P__inference_dense_activation_1_layer_call_and_return_conditional_losses_139790602$
"dense_activation_1/PartitionedCall╦
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_13978852*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absй
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul╤
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_13978825*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareй
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const┌
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul╦
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_13978798*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absй
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul░
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_13979041*
_output_shapes
:	└T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└T2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const╖
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЖ
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

IdentityЗ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_1_conv_1/StatefulPartitionedCall9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         ▄:         ▄:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:         ▄
 
_user_specified_nameinputs:TP
,
_output_shapes
:         ▄
 
_user_specified_nameinputs:TP
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
╚
k
2__inference_stream_0_drop_1_layer_call_fn_13982374

inputs
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_139792122
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
Д
╫
M__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_13981818

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄2
conv1d/ExpandDims╕
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
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2	
BiasAddф
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareй
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const┌
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identity╟
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▄: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
Р∙
╢ 
E__inference_model_1_layer_call_and_return_conditional_losses_13981130

inputs[
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
0basemodel_dense_1_matmul_readvariableop_resource:	└T?
1basemodel_dense_1_biasadd_readvariableop_resource:TU
Gbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource:TW
Ibasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource:TS
Ebasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:TO
Abasemodel_batch_normalization_3_batchnorm_readvariableop_resource:T
identityИв-basemodel/batch_normalization/AssignMovingAvgв<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOpв/basemodel/batch_normalization/AssignMovingAvg_1в>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOpв6basemodel/batch_normalization/batchnorm/ReadVariableOpв:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpв/basemodel/batch_normalization_1/AssignMovingAvgв>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOpв1basemodel/batch_normalization_1/AssignMovingAvg_1в@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpв8basemodel/batch_normalization_1/batchnorm/ReadVariableOpв<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpв/basemodel/batch_normalization_2/AssignMovingAvgв>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOpв1basemodel/batch_normalization_2/AssignMovingAvg_1в@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpв8basemodel/batch_normalization_2/batchnorm/ReadVariableOpв<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpв/basemodel/batch_normalization_3/AssignMovingAvgв>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOpв1basemodel/batch_normalization_3/AssignMovingAvg_1в@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpв8basemodel/batch_normalization_3/batchnorm/ReadVariableOpв<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpв(basemodel/dense_1/BiasAdd/ReadVariableOpв'basemodel/dense_1/MatMul/ReadVariableOpв0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpв<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpв0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpв<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpв0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpв<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЯ
+basemodel/stream_2_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2-
+basemodel/stream_2_input_drop/dropout/Const╥
)basemodel/stream_2_input_drop/dropout/MulMulinputs4basemodel/stream_2_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:         ▄2+
)basemodel/stream_2_input_drop/dropout/MulР
+basemodel/stream_2_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2-
+basemodel/stream_2_input_drop/dropout/Shapeо
Bbasemodel/stream_2_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_2_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:         ▄*
dtype0*
seed╖*
seed2╖2D
Bbasemodel/stream_2_input_drop/dropout/random_uniform/RandomUniform▒
4basemodel/stream_2_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>26
4basemodel/stream_2_input_drop/dropout/GreaterEqual/y╗
2basemodel/stream_2_input_drop/dropout/GreaterEqualGreaterEqualKbasemodel/stream_2_input_drop/dropout/random_uniform/RandomUniform:output:0=basemodel/stream_2_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄24
2basemodel/stream_2_input_drop/dropout/GreaterEqual▐
*basemodel/stream_2_input_drop/dropout/CastCast6basemodel/stream_2_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄2,
*basemodel/stream_2_input_drop/dropout/Castў
+basemodel/stream_2_input_drop/dropout/Mul_1Mul-basemodel/stream_2_input_drop/dropout/Mul:z:0.basemodel/stream_2_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄2-
+basemodel/stream_2_input_drop/dropout/Mul_1Я
+basemodel/stream_1_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2-
+basemodel/stream_1_input_drop/dropout/Const╥
)basemodel/stream_1_input_drop/dropout/MulMulinputs4basemodel/stream_1_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:         ▄2+
)basemodel/stream_1_input_drop/dropout/MulР
+basemodel/stream_1_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2-
+basemodel/stream_1_input_drop/dropout/Shapeо
Bbasemodel/stream_1_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_1_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:         ▄*
dtype0*
seed╖*
seed2╖2D
Bbasemodel/stream_1_input_drop/dropout/random_uniform/RandomUniform▒
4basemodel/stream_1_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>26
4basemodel/stream_1_input_drop/dropout/GreaterEqual/y╗
2basemodel/stream_1_input_drop/dropout/GreaterEqualGreaterEqualKbasemodel/stream_1_input_drop/dropout/random_uniform/RandomUniform:output:0=basemodel/stream_1_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄24
2basemodel/stream_1_input_drop/dropout/GreaterEqual▐
*basemodel/stream_1_input_drop/dropout/CastCast6basemodel/stream_1_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄2,
*basemodel/stream_1_input_drop/dropout/Castў
+basemodel/stream_1_input_drop/dropout/Mul_1Mul-basemodel/stream_1_input_drop/dropout/Mul:z:0.basemodel/stream_1_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄2-
+basemodel/stream_1_input_drop/dropout/Mul_1Я
+basemodel/stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2-
+basemodel/stream_0_input_drop/dropout/Const╥
)basemodel/stream_0_input_drop/dropout/MulMulinputs4basemodel/stream_0_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:         ▄2+
)basemodel/stream_0_input_drop/dropout/MulР
+basemodel/stream_0_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2-
+basemodel/stream_0_input_drop/dropout/Shapeо
Bbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:         ▄*
dtype0*
seed╖*
seed2╖2D
Bbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniform▒
4basemodel/stream_0_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>26
4basemodel/stream_0_input_drop/dropout/GreaterEqual/y╗
2basemodel/stream_0_input_drop/dropout/GreaterEqualGreaterEqualKbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniform:output:0=basemodel/stream_0_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄24
2basemodel/stream_0_input_drop/dropout/GreaterEqual▐
*basemodel/stream_0_input_drop/dropout/CastCast6basemodel/stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄2,
*basemodel/stream_0_input_drop/dropout/Castў
+basemodel/stream_0_input_drop/dropout/Mul_1Mul-basemodel/stream_0_input_drop/dropout/Mul:z:0.basemodel/stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄2-
+basemodel/stream_0_input_drop/dropout/Mul_1н
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        21
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimО
+basemodel/stream_2_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_2_input_drop/dropout/Mul_1:z:08basemodel/stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄2-
+basemodel/stream_2_conv_1/conv1d/ExpandDimsЖ
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpи
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
-basemodel/stream_2_conv_1/conv1d/ExpandDims_1Я
 basemodel/stream_2_conv_1/conv1dConv2D4basemodel/stream_2_conv_1/conv1d/ExpandDims:output:06basemodel/stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2"
 basemodel/stream_2_conv_1/conv1dс
(basemodel/stream_2_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        2*
(basemodel/stream_2_conv_1/conv1d/Squeeze┌
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpї
!basemodel/stream_2_conv_1/BiasAddBiasAdd1basemodel/stream_2_conv_1/conv1d/Squeeze:output:08basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2#
!basemodel/stream_2_conv_1/BiasAddн
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        21
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimО
+basemodel/stream_1_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_1_input_drop/dropout/Mul_1:z:08basemodel/stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄2-
+basemodel/stream_1_conv_1/conv1d/ExpandDimsЖ
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpи
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
-basemodel/stream_1_conv_1/conv1d/ExpandDims_1Я
 basemodel/stream_1_conv_1/conv1dConv2D4basemodel/stream_1_conv_1/conv1d/ExpandDims:output:06basemodel/stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2"
 basemodel/stream_1_conv_1/conv1dс
(basemodel/stream_1_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        2*
(basemodel/stream_1_conv_1/conv1d/Squeeze┌
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpї
!basemodel/stream_1_conv_1/BiasAddBiasAdd1basemodel/stream_1_conv_1/conv1d/Squeeze:output:08basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2#
!basemodel/stream_1_conv_1/BiasAddн
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        21
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimО
+basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_0_input_drop/dropout/Mul_1:z:08basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄2-
+basemodel/stream_0_conv_1/conv1d/ExpandDimsЖ
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpи
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
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1Я
 basemodel/stream_0_conv_1/conv1dConv2D4basemodel/stream_0_conv_1/conv1d/ExpandDims:output:06basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1dс
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        2*
(basemodel/stream_0_conv_1/conv1d/Squeeze┌
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpї
!basemodel/stream_0_conv_1/BiasAddBiasAdd1basemodel/stream_0_conv_1/conv1d/Squeeze:output:08basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2#
!basemodel/stream_0_conv_1/BiasAdd╤
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
,basemodel/batch_normalization_2/moments/meanр
4basemodel/batch_normalization_2/moments/StopGradientStopGradient5basemodel/batch_normalization_2/moments/mean:output:0*
T0*"
_output_shapes
:@26
4basemodel/batch_normalization_2/moments/StopGradientн
9basemodel/batch_normalization_2/moments/SquaredDifferenceSquaredDifference*basemodel/stream_2_conv_1/BiasAdd:output:0=basemodel/batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:         ▄@2;
9basemodel/batch_normalization_2/moments/SquaredDifference┘
Bbasemodel/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bbasemodel/batch_normalization_2/moments/variance/reduction_indices╢
0basemodel/batch_normalization_2/moments/varianceMean=basemodel/batch_normalization_2/moments/SquaredDifference:z:0Kbasemodel/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(22
0basemodel/batch_normalization_2/moments/varianceс
/basemodel/batch_normalization_2/moments/SqueezeSqueeze5basemodel/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 21
/basemodel/batch_normalization_2/moments/Squeezeщ
1basemodel/batch_normalization_2/moments/Squeeze_1Squeeze9basemodel/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 23
1basemodel/batch_normalization_2/moments/Squeeze_1│
5basemodel/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<27
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
3basemodel/batch_normalization_2/AssignMovingAvg/mul▀
/basemodel/batch_normalization_2/AssignMovingAvgAssignSubVariableOpGbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource7basemodel/batch_normalization_2/AssignMovingAvg/mul:z:0?^basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization_2/AssignMovingAvg╖
7basemodel/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<29
7basemodel/batch_normalization_2/AssignMovingAvg_1/decayК
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpа
5basemodel/batch_normalization_2/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_2/AssignMovingAvg_1/subЧ
5basemodel/batch_normalization_2/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_2/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_2/AssignMovingAvg_1/mulщ
1basemodel/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_2/AssignMovingAvg_1/mul:z:0A^basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_2/AssignMovingAvg_1з
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
-basemodel/batch_normalization_2/batchnorm/add├
/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm/Rsqrt■
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpЕ
-basemodel/batch_normalization_2/batchnorm/mulMul3basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/mul 
/basemodel/batch_normalization_2/batchnorm/mul_1Mul*basemodel/stream_2_conv_1/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@21
/basemodel/batch_normalization_2/batchnorm/mul_1√
/basemodel/batch_normalization_2/batchnorm/mul_2Mul8basemodel/batch_normalization_2/moments/Squeeze:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm/mul_2Є
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpБ
-basemodel/batch_normalization_2/batchnorm/subSub@basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/subК
/basemodel/batch_normalization_2/batchnorm/add_1AddV23basemodel/batch_normalization_2/batchnorm/mul_1:z:01basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@21
/basemodel/batch_normalization_2/batchnorm/add_1╤
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
,basemodel/batch_normalization_1/moments/meanр
4basemodel/batch_normalization_1/moments/StopGradientStopGradient5basemodel/batch_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:@26
4basemodel/batch_normalization_1/moments/StopGradientн
9basemodel/batch_normalization_1/moments/SquaredDifferenceSquaredDifference*basemodel/stream_1_conv_1/BiasAdd:output:0=basemodel/batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:         ▄@2;
9basemodel/batch_normalization_1/moments/SquaredDifference┘
Bbasemodel/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bbasemodel/batch_normalization_1/moments/variance/reduction_indices╢
0basemodel/batch_normalization_1/moments/varianceMean=basemodel/batch_normalization_1/moments/SquaredDifference:z:0Kbasemodel/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(22
0basemodel/batch_normalization_1/moments/varianceс
/basemodel/batch_normalization_1/moments/SqueezeSqueeze5basemodel/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 21
/basemodel/batch_normalization_1/moments/Squeezeщ
1basemodel/batch_normalization_1/moments/Squeeze_1Squeeze9basemodel/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 23
1basemodel/batch_normalization_1/moments/Squeeze_1│
5basemodel/batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<27
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
3basemodel/batch_normalization_1/AssignMovingAvg/mul▀
/basemodel/batch_normalization_1/AssignMovingAvgAssignSubVariableOpGbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource7basemodel/batch_normalization_1/AssignMovingAvg/mul:z:0?^basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization_1/AssignMovingAvg╖
7basemodel/batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<29
7basemodel/batch_normalization_1/AssignMovingAvg_1/decayК
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpа
5basemodel/batch_normalization_1/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_1/AssignMovingAvg_1/subЧ
5basemodel/batch_normalization_1/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_1/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_1/AssignMovingAvg_1/mulщ
1basemodel/batch_normalization_1/AssignMovingAvg_1AssignSubVariableOpIbasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_1/AssignMovingAvg_1/mul:z:0A^basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_1/AssignMovingAvg_1з
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
-basemodel/batch_normalization_1/batchnorm/add├
/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm/Rsqrt■
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpЕ
-basemodel/batch_normalization_1/batchnorm/mulMul3basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/mul 
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_1_conv_1/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@21
/basemodel/batch_normalization_1/batchnorm/mul_1√
/basemodel/batch_normalization_1/batchnorm/mul_2Mul8basemodel/batch_normalization_1/moments/Squeeze:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm/mul_2Є
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpБ
-basemodel/batch_normalization_1/batchnorm/subSub@basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/subК
/basemodel/batch_normalization_1/batchnorm/add_1AddV23basemodel/batch_normalization_1/batchnorm/mul_1:z:01basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@21
/basemodel/batch_normalization_1/batchnorm/add_1═
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
*basemodel/batch_normalization/moments/mean┌
2basemodel/batch_normalization/moments/StopGradientStopGradient3basemodel/batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:@24
2basemodel/batch_normalization/moments/StopGradientз
7basemodel/batch_normalization/moments/SquaredDifferenceSquaredDifference*basemodel/stream_0_conv_1/BiasAdd:output:0;basemodel/batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:         ▄@29
7basemodel/batch_normalization/moments/SquaredDifference╒
@basemodel/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2B
@basemodel/batch_normalization/moments/variance/reduction_indicesо
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
-basemodel/batch_normalization/moments/Squeezeу
/basemodel/batch_normalization/moments/Squeeze_1Squeeze7basemodel/batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 21
/basemodel/batch_normalization/moments/Squeeze_1п
3basemodel/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<25
3basemodel/batch_normalization/AssignMovingAvg/decay■
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
1basemodel/batch_normalization/AssignMovingAvg/mul╒
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
╫#<27
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
3basemodel/batch_normalization/AssignMovingAvg_1/mul▀
/basemodel/batch_normalization/AssignMovingAvg_1AssignSubVariableOpGbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource7basemodel/batch_normalization/AssignMovingAvg_1/mul:z:0?^basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization/AssignMovingAvg_1г
-basemodel/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2/
-basemodel/batch_normalization/batchnorm/add/y·
+basemodel/batch_normalization/batchnorm/addAddV28basemodel/batch_normalization/moments/Squeeze_1:output:06basemodel/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/add╜
-basemodel/batch_normalization/batchnorm/RsqrtRsqrt/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/Rsqrt°
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp¤
+basemodel/batch_normalization/batchnorm/mulMul1basemodel/batch_normalization/batchnorm/Rsqrt:y:0Bbasemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/mul∙
-basemodel/batch_normalization/batchnorm/mul_1Mul*basemodel/stream_0_conv_1/BiasAdd:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2/
-basemodel/batch_normalization/batchnorm/mul_1є
-basemodel/batch_normalization/batchnorm/mul_2Mul6basemodel/batch_normalization/moments/Squeeze:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/mul_2ь
6basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype028
6basemodel/batch_normalization/batchnorm/ReadVariableOp∙
+basemodel/batch_normalization/batchnorm/subSub>basemodel/batch_normalization/batchnorm/ReadVariableOp:value:01basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/subВ
-basemodel/batch_normalization/batchnorm/add_1AddV21basemodel/batch_normalization/batchnorm/mul_1:z:0/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2/
-basemodel/batch_normalization/batchnorm/add_1о
basemodel/activation_2/ReluRelu3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ▄@2
basemodel/activation_2/Reluо
basemodel/activation_1/ReluRelu3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ▄@2
basemodel/activation_1/Reluи
basemodel/activation/ReluRelu1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ▄@2
basemodel/activation/ReluЧ
'basemodel/stream_2_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2)
'basemodel/stream_2_drop_1/dropout/Constщ
%basemodel/stream_2_drop_1/dropout/MulMul)basemodel/activation_2/Relu:activations:00basemodel/stream_2_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:         ▄@2'
%basemodel/stream_2_drop_1/dropout/Mulл
'basemodel/stream_2_drop_1/dropout/ShapeShape)basemodel/activation_2/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_2_drop_1/dropout/Shapeв
>basemodel/stream_2_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_2_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:         ▄@*
dtype0*
seed╖*
seed2╖2@
>basemodel/stream_2_drop_1/dropout/random_uniform/RandomUniformй
0basemodel/stream_2_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>22
0basemodel/stream_2_drop_1/dropout/GreaterEqual/yл
.basemodel/stream_2_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_2_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_2_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄@20
.basemodel/stream_2_drop_1/dropout/GreaterEqual╥
&basemodel/stream_2_drop_1/dropout/CastCast2basemodel/stream_2_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄@2(
&basemodel/stream_2_drop_1/dropout/Castч
'basemodel/stream_2_drop_1/dropout/Mul_1Mul)basemodel/stream_2_drop_1/dropout/Mul:z:0*basemodel/stream_2_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄@2)
'basemodel/stream_2_drop_1/dropout/Mul_1Ч
'basemodel/stream_1_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2)
'basemodel/stream_1_drop_1/dropout/Constщ
%basemodel/stream_1_drop_1/dropout/MulMul)basemodel/activation_1/Relu:activations:00basemodel/stream_1_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:         ▄@2'
%basemodel/stream_1_drop_1/dropout/Mulл
'basemodel/stream_1_drop_1/dropout/ShapeShape)basemodel/activation_1/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_1_drop_1/dropout/Shapeв
>basemodel/stream_1_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_1_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:         ▄@*
dtype0*
seed╖*
seed2╖2@
>basemodel/stream_1_drop_1/dropout/random_uniform/RandomUniformй
0basemodel/stream_1_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>22
0basemodel/stream_1_drop_1/dropout/GreaterEqual/yл
.basemodel/stream_1_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_1_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_1_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄@20
.basemodel/stream_1_drop_1/dropout/GreaterEqual╥
&basemodel/stream_1_drop_1/dropout/CastCast2basemodel/stream_1_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄@2(
&basemodel/stream_1_drop_1/dropout/Castч
'basemodel/stream_1_drop_1/dropout/Mul_1Mul)basemodel/stream_1_drop_1/dropout/Mul:z:0*basemodel/stream_1_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄@2)
'basemodel/stream_1_drop_1/dropout/Mul_1Ч
'basemodel/stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2)
'basemodel/stream_0_drop_1/dropout/Constч
%basemodel/stream_0_drop_1/dropout/MulMul'basemodel/activation/Relu:activations:00basemodel/stream_0_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:         ▄@2'
%basemodel/stream_0_drop_1/dropout/Mulй
'basemodel/stream_0_drop_1/dropout/ShapeShape'basemodel/activation/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_1/dropout/Shapeв
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:         ▄@*
dtype0*
seed╖*
seed2╖2@
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniformй
0basemodel/stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>22
0basemodel/stream_0_drop_1/dropout/GreaterEqual/yл
.basemodel/stream_0_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_0_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄@20
.basemodel/stream_0_drop_1/dropout/GreaterEqual╥
&basemodel/stream_0_drop_1/dropout/CastCast2basemodel/stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄@2(
&basemodel/stream_0_drop_1/dropout/Castч
'basemodel/stream_0_drop_1/dropout/Mul_1Mul)basemodel/stream_0_drop_1/dropout/Mul:z:0*basemodel/stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄@2)
'basemodel/stream_0_drop_1/dropout/Mul_1╕
9basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9basemodel/global_average_pooling1d/Mean/reduction_indices¤
'basemodel/global_average_pooling1d/MeanMean+basemodel/stream_0_drop_1/dropout/Mul_1:z:0Bbasemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2)
'basemodel/global_average_pooling1d/Mean╝
;basemodel/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d_1/Mean/reduction_indicesГ
)basemodel/global_average_pooling1d_1/MeanMean+basemodel/stream_1_drop_1/dropout/Mul_1:z:0Dbasemodel/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2+
)basemodel/global_average_pooling1d_1/Mean╝
;basemodel/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d_2/Mean/reduction_indicesГ
)basemodel/global_average_pooling1d_2/MeanMean+basemodel/stream_2_drop_1/dropout/Mul_1:z:0Dbasemodel/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2+
)basemodel/global_average_pooling1d_2/MeanИ
!basemodel/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!basemodel/concatenate/concat/axis╩
basemodel/concatenate/concatConcatV20basemodel/global_average_pooling1d/Mean:output:02basemodel/global_average_pooling1d_1/Mean:output:02basemodel/global_average_pooling1d_2/Mean:output:0*basemodel/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:         └2
basemodel/concatenate/concat─
'basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	└T*
dtype02)
'basemodel/dense_1/MatMul/ReadVariableOp╚
basemodel/dense_1/MatMulMatMul%basemodel/concatenate/concat:output:0/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
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
>basemodel/batch_normalization_3/moments/mean/reduction_indicesЛ
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
9basemodel/batch_normalization_3/moments/SquaredDifference╥
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
0basemodel/batch_normalization_3/moments/varianceр
/basemodel/batch_normalization_3/moments/SqueezeSqueeze5basemodel/batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 21
/basemodel/batch_normalization_3/moments/Squeezeш
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
╫#<27
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
3basemodel/batch_normalization_3/AssignMovingAvg/mul▀
/basemodel/batch_normalization_3/AssignMovingAvgAssignSubVariableOpGbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource7basemodel/batch_normalization_3/AssignMovingAvg/mul:z:0?^basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization_3/AssignMovingAvg╖
7basemodel/batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<29
7basemodel/batch_normalization_3/AssignMovingAvg_1/decayК
@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:T*
dtype02B
@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpа
5basemodel/batch_normalization_3/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_1/subЧ
5basemodel/batch_normalization_3/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_3/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_3/AssignMovingAvg_1/mulщ
1basemodel/batch_normalization_3/AssignMovingAvg_1AssignSubVariableOpIbasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_3/AssignMovingAvg_1/mul:z:0A^basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_3/AssignMovingAvg_1з
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
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpЕ
-basemodel/batch_normalization_3/batchnorm/mulMul3basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_3/batchnorm/mulЄ
/basemodel/batch_normalization_3/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T21
/basemodel/batch_normalization_3/batchnorm/mul_1√
/basemodel/batch_normalization_3/batchnorm/mul_2Mul8basemodel/batch_normalization_3/moments/Squeeze:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_3/batchnorm/mul_2Є
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
:         T21
/basemodel/batch_normalization_3/batchnorm/add_1╛
$basemodel/dense_activation_1/SigmoidSigmoid3basemodel/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2&
$basemodel/dense_activation_1/Sigmoid°
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absй
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul■
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareй
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const┌
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul°
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absй
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul╨
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	└T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└T2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const╖
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulГ
IdentityIdentity(basemodel/dense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         T2

Identity╪
NoOpNoOp.^basemodel/batch_normalization/AssignMovingAvg=^basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_1?^basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_1/AssignMovingAvg?^basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_1A^basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_2/AssignMovingAvg?^basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_1A^basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_3/AssignMovingAvg?^basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_3/AssignMovingAvg_1A^basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_3/batchnorm/ReadVariableOp=^basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 2^
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
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp2|
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp2|
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
Т
o
Q__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_13978774

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ▄2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ▄2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
л:
В
!__inference__traced_save_13982805
file_prefix5
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
@savev2_batch_normalization_3_moving_variance_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameН

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Я	
valueХ	BТ	B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names║
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЖ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_stream_0_conv_1_kernel_read_readvariableop/savev2_stream_0_conv_1_bias_read_readvariableop1savev2_stream_1_conv_1_kernel_read_readvariableop/savev2_stream_1_conv_1_bias_read_readvariableop1savev2_stream_2_conv_1_kernel_read_readvariableop/savev2_stream_2_conv_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*╞
_input_shapes┤
▒: :@:@:@:@:@:@:@:@:@:@:@:@:	└T:T:T:T:@:@:@:@:@:@:T:T: 2(
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
:@:($
"
_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	└T: 

_output_shapes
:T: 

_output_shapes
:T: 

_output_shapes
:T: 
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
:T: 

_output_shapes
:T:

_output_shapes
: 
╒
N
2__inference_dense_1_dropout_layer_call_fn_13982531

inputs
identity╧
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_dense_1_dropout_layer_call_and_return_conditional_losses_139790222
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         └:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
Й+
ь
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_13982174

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
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
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         ▄@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
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
╫#<2
AssignMovingAvg/decayд
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
╫#<2
AssignMovingAvg_1/decayк
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2
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
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ▄@2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▄@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
╣
▓
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13982246

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
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
 :                  @2
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
Т
o
Q__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_13981707

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ▄2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ▄2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
║+
ь
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13982280

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
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
moments/StopGradient▒
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                  @2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
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
╫#<2
AssignMovingAvg/decayд
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
╫#<2
AssignMovingAvg_1/decayк
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
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
 :                  @2
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
 :                  @2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                  @2

IdentityЄ
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
э
W
;__inference_global_average_pooling1d_layer_call_fn_13982455

inputs
identity╫
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_139789912
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
е
Y
=__inference_global_average_pooling1d_1_layer_call_fn_13982472

inputs
identityт
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
GPU2*0J 8В *a
f\RZ
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_139785442
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
■
м
__inference_loss_fn_3_13982710I
6dense_1_kernel_regularizer_abs_readvariableop_resource:	└T
identityИв-dense_1/kernel/Regularizer/Abs/ReadVariableOp╓
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	└T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└T2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const╖
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
╫#<2"
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
Й+
ь
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_13979388

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
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
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         ▄@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
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
╫#<2
AssignMovingAvg/decayд
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
╫#<2
AssignMovingAvg_1/decayк
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2
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
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ▄@2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▄@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
э
R
6__inference_stream_0_input_drop_layer_call_fn_13981670

inputs
identity╫
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_139787742
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ▄2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
Р
▓
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_13982140

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2
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
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▄@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
З+
ъ
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_13982014

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
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
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         ▄@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
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
╫#<2
AssignMovingAvg/decayд
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
╫#<2
AssignMovingAvg_1/decayк
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/subЧ
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2
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
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ▄@2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▄@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
Ў
f
J__inference_activation_2_layer_call_and_return_conditional_losses_13978949

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         ▄@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         ▄@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
ё
Y
=__inference_global_average_pooling1d_1_layer_call_fn_13982477

inputs
identity┘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *a
f\RZ
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_139789982
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
Ў
f
J__inference_activation_1_layer_call_and_return_conditional_losses_13978956

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         ▄@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         ▄@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
яб
ъ
G__inference_basemodel_layer_call_and_return_conditional_losses_13980026
inputs_0
inputs_1
inputs_2.
stream_2_conv_1_13979933:@&
stream_2_conv_1_13979935:@.
stream_1_conv_1_13979938:@&
stream_1_conv_1_13979940:@.
stream_0_conv_1_13979943:@&
stream_0_conv_1_13979945:@,
batch_normalization_2_13979948:@,
batch_normalization_2_13979950:@,
batch_normalization_2_13979952:@,
batch_normalization_2_13979954:@,
batch_normalization_1_13979957:@,
batch_normalization_1_13979959:@,
batch_normalization_1_13979961:@,
batch_normalization_1_13979963:@*
batch_normalization_13979966:@*
batch_normalization_13979968:@*
batch_normalization_13979970:@*
batch_normalization_13979972:@#
dense_1_13979986:	└T
dense_1_13979988:T,
batch_normalization_3_13979991:T,
batch_normalization_3_13979993:T,
batch_normalization_3_13979995:T,
batch_normalization_3_13979997:T
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв'stream_0_conv_1/StatefulPartitionedCallв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв'stream_0_drop_1/StatefulPartitionedCallв+stream_0_input_drop/StatefulPartitionedCallв'stream_1_conv_1/StatefulPartitionedCallв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв'stream_1_drop_1/StatefulPartitionedCallв+stream_1_input_drop/StatefulPartitionedCallв'stream_2_conv_1/StatefulPartitionedCallв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpв'stream_2_drop_1/StatefulPartitionedCallв+stream_2_input_drop/StatefulPartitionedCallЩ
+stream_2_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_139795552-
+stream_2_input_drop/StatefulPartitionedCall╟
+stream_1_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_1,^stream_2_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_139795322-
+stream_1_input_drop/StatefulPartitionedCall╟
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_0,^stream_1_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_139795092-
+stream_0_input_drop/StatefulPartitionedCallє
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_2_input_drop/StatefulPartitionedCall:output:0stream_2_conv_1_13979933stream_2_conv_1_13979935*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_139787972)
'stream_2_conv_1/StatefulPartitionedCallє
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_1_input_drop/StatefulPartitionedCall:output:0stream_1_conv_1_13979938stream_1_conv_1_13979940*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_139788242)
'stream_1_conv_1/StatefulPartitionedCallє
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_13979943stream_0_conv_1_13979945*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_139788512)
'stream_0_conv_1/StatefulPartitionedCall╧
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_13979948batch_normalization_2_13979950batch_normalization_2_13979952batch_normalization_2_13979954*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_139794482/
-batch_normalization_2/StatefulPartitionedCall╧
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_13979957batch_normalization_1_13979959batch_normalization_1_13979961batch_normalization_1_13979963*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_139793882/
-batch_normalization_1/StatefulPartitionedCall┴
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_13979966batch_normalization_13979968batch_normalization_13979970batch_normalization_13979972*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_139793282-
+batch_normalization/StatefulPartitionedCallЪ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_activation_2_layer_call_and_return_conditional_losses_139789492
activation_2/PartitionedCallЪ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_activation_1_layer_call_and_return_conditional_losses_139789562
activation_1/PartitionedCallТ
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_139789632
activation/PartitionedCall╪
'stream_2_drop_1/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_139792582)
'stream_2_drop_1/StatefulPartitionedCall╘
'stream_1_drop_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0(^stream_2_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_139792352)
'stream_1_drop_1/StatefulPartitionedCall╥
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0(^stream_1_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_139792122)
'stream_0_drop_1/StatefulPartitionedCall│
(global_average_pooling1d/PartitionedCallPartitionedCall0stream_0_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_139789912*
(global_average_pooling1d/PartitionedCall╣
*global_average_pooling1d_1/PartitionedCallPartitionedCall0stream_1_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *a
f\RZ
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_139789982,
*global_average_pooling1d_1/PartitionedCall╣
*global_average_pooling1d_2/PartitionedCallPartitionedCall0stream_2_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *a
f\RZ
X__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_139790052,
*global_average_pooling1d_2/PartitionedCall·
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:03global_average_pooling1d_1/PartitionedCall:output:03global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_139790152
concatenate/PartitionedCallН
dense_1_dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_dense_1_dropout_layer_call_and_return_conditional_losses_139791662!
dense_1_dropout/PartitionedCall║
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_13979986dense_1_13979988*
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
GPU2*0J 8В *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_139790402!
dense_1/StatefulPartitionedCall┬
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_13979991batch_normalization_3_13979993batch_normalization_3_13979995batch_normalization_3_13979997*
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
GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_139786662/
-batch_normalization_3/StatefulPartitionedCallз
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
GPU2*0J 8В *Y
fTRR
P__inference_dense_activation_1_layer_call_and_return_conditional_losses_139790602$
"dense_activation_1/PartitionedCall╦
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_13979943*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absй
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul╤
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_13979938*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareй
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const┌
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul╦
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_13979933*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absй
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul░
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_13979986*
_output_shapes
:	└T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└T2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const╖
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЖ
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

IdentityП
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall(^stream_1_conv_1/StatefulPartitionedCall9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_1_drop_1/StatefulPartitionedCall,^stream_1_input_drop/StatefulPartitionedCall(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_2_drop_1/StatefulPartitionedCall,^stream_2_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         ▄:         ▄:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:         ▄
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:         ▄
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:         ▄
"
_user_specified_name
inputs_2
╨
o
6__inference_stream_2_input_drop_layer_call_fn_13981729

inputs
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_139795552
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ▄2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
╧
h
.__inference_concatenate_layer_call_fn_13982518
inputs_0
inputs_1
inputs_2
identityу
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_139790152
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         @:         @:         @:Q M
'
_output_shapes
:         @
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         @
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:         @
"
_user_specified_name
inputs/2
ў
l
M__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_13979212

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ▄@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ▄@*
dtype0*
seed╖*
seed2╖2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ▄@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
Р
▓
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13982300

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@2
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
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▄@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
╝
r
V__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_13982461

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
є
┼
__inference_loss_fn_1_13982688W
Astream_1_conv_1_kernel_regularizer_square_readvariableop_resource:@
identityИв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp·
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAstream_1_conv_1_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareй
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const┌
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
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
И
r
V__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_13982467

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
:         @2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
Н	
╤
6__inference_batch_normalization_layer_call_fn_13981880

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallй
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
GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_139781082
StatefulPartitionedCallИ
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
╛
t
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_13982483

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
б
W
;__inference_global_average_pooling1d_layer_call_fn_13982450

inputs
identityр
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
GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_139785202
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
ъ
l
P__inference_dense_activation_1_layer_call_and_return_conditional_losses_13979060

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
▐
╙
8__inference_batch_normalization_3_layer_call_fn_13982589

inputs
unknown:T
	unknown_0:T
	unknown_1:T
	unknown_2:T
identityИвStatefulPartitionedCallа
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
GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_139786062
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
ьМ
┌
#__inference__wrapped_model_13978024
left_inputsc
Mmodel_1_basemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource:@O
Amodel_1_basemodel_stream_2_conv_1_biasadd_readvariableop_resource:@c
Mmodel_1_basemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource:@O
Amodel_1_basemodel_stream_1_conv_1_biasadd_readvariableop_resource:@c
Mmodel_1_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource:@O
Amodel_1_basemodel_stream_0_conv_1_biasadd_readvariableop_resource:@W
Imodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_resource:@[
Mmodel_1_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource:@Y
Kmodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource:@Y
Kmodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource:@W
Imodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_resource:@[
Mmodel_1_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:@Y
Kmodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource:@Y
Kmodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource:@U
Gmodel_1_basemodel_batch_normalization_batchnorm_readvariableop_resource:@Y
Kmodel_1_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource:@W
Imodel_1_basemodel_batch_normalization_batchnorm_readvariableop_1_resource:@W
Imodel_1_basemodel_batch_normalization_batchnorm_readvariableop_2_resource:@K
8model_1_basemodel_dense_1_matmul_readvariableop_resource:	└TG
9model_1_basemodel_dense_1_biasadd_readvariableop_resource:TW
Imodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_resource:T[
Mmodel_1_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:TY
Kmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource:TY
Kmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource:T
identityИв>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOpв@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1в@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2вBmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpв@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpвBmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1вBmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2вDmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpв@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOpвBmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1вBmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2вDmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpв@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOpвBmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1вBmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2вDmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpв0model_1/basemodel/dense_1/BiasAdd/ReadVariableOpв/model_1/basemodel/dense_1/MatMul/ReadVariableOpв8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpвDmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpв8model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpвDmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpв8model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpвDmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp░
.model_1/basemodel/stream_2_input_drop/IdentityIdentityleft_inputs*
T0*,
_output_shapes
:         ▄20
.model_1/basemodel/stream_2_input_drop/Identity░
.model_1/basemodel/stream_1_input_drop/IdentityIdentityleft_inputs*
T0*,
_output_shapes
:         ▄20
.model_1/basemodel/stream_1_input_drop/Identity░
.model_1/basemodel/stream_0_input_drop/IdentityIdentityleft_inputs*
T0*,
_output_shapes
:         ▄20
.model_1/basemodel/stream_0_input_drop/Identity╜
7model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        29
7model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimо
3model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims
ExpandDims7model_1/basemodel/stream_2_input_drop/Identity:output:0@model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄25
3model_1/basemodel/stream_2_conv_1/conv1d/ExpandDimsЮ
Dmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpMmodel_1_basemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp╕
9model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dim┐
5model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1
ExpandDimsLmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Bmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@27
5model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1┐
(model_1/basemodel/stream_2_conv_1/conv1dConv2D<model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims:output:0>model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2*
(model_1/basemodel/stream_2_conv_1/conv1d∙
0model_1/basemodel/stream_2_conv_1/conv1d/SqueezeSqueeze1model_1/basemodel/stream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        22
0model_1/basemodel/stream_2_conv_1/conv1d/SqueezeЄ
8model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOpAmodel_1_basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpХ
)model_1/basemodel/stream_2_conv_1/BiasAddBiasAdd9model_1/basemodel/stream_2_conv_1/conv1d/Squeeze:output:0@model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2+
)model_1/basemodel/stream_2_conv_1/BiasAdd╜
7model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        29
7model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimо
3model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims
ExpandDims7model_1/basemodel/stream_1_input_drop/Identity:output:0@model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄25
3model_1/basemodel/stream_1_conv_1/conv1d/ExpandDimsЮ
Dmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpMmodel_1_basemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp╕
9model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dim┐
5model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1
ExpandDimsLmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Bmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@27
5model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1┐
(model_1/basemodel/stream_1_conv_1/conv1dConv2D<model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims:output:0>model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2*
(model_1/basemodel/stream_1_conv_1/conv1d∙
0model_1/basemodel/stream_1_conv_1/conv1d/SqueezeSqueeze1model_1/basemodel/stream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        22
0model_1/basemodel/stream_1_conv_1/conv1d/SqueezeЄ
8model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOpAmodel_1_basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpХ
)model_1/basemodel/stream_1_conv_1/BiasAddBiasAdd9model_1/basemodel/stream_1_conv_1/conv1d/Squeeze:output:0@model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2+
)model_1/basemodel/stream_1_conv_1/BiasAdd╜
7model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        29
7model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimо
3model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims7model_1/basemodel/stream_0_input_drop/Identity:output:0@model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄25
3model_1/basemodel/stream_0_conv_1/conv1d/ExpandDimsЮ
Dmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpMmodel_1_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp╕
9model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim┐
5model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1
ExpandDimsLmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Bmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@27
5model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1┐
(model_1/basemodel/stream_0_conv_1/conv1dConv2D<model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims:output:0>model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2*
(model_1/basemodel/stream_0_conv_1/conv1d∙
0model_1/basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze1model_1/basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        22
0model_1/basemodel/stream_0_conv_1/conv1d/SqueezeЄ
8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOpAmodel_1_basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpХ
)model_1/basemodel/stream_0_conv_1/BiasAddBiasAdd9model_1/basemodel/stream_0_conv_1/conv1d/Squeeze:output:0@model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2+
)model_1/basemodel/stream_0_conv_1/BiasAddК
@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpImodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02B
@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp╖
7model_1/basemodel/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:29
7model_1/basemodel/batch_normalization_2/batchnorm/add/yи
5model_1/basemodel/batch_normalization_2/batchnorm/addAddV2Hmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:0@model_1/basemodel/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_2/batchnorm/add█
7model_1/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt9model_1/basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@29
7model_1/basemodel/batch_normalization_2/batchnorm/RsqrtЦ
Dmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpMmodel_1_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpе
5model_1/basemodel/batch_normalization_2/batchnorm/mulMul;model_1/basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Lmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_2/batchnorm/mulЯ
7model_1/basemodel/batch_normalization_2/batchnorm/mul_1Mul2model_1/basemodel/stream_2_conv_1/BiasAdd:output:09model_1/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@29
7model_1/basemodel/batch_normalization_2/batchnorm/mul_1Р
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpKmodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1е
7model_1/basemodel/batch_normalization_2/batchnorm/mul_2MulJmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:value:09model_1/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@29
7model_1/basemodel/batch_normalization_2/batchnorm/mul_2Р
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpKmodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2г
5model_1/basemodel/batch_normalization_2/batchnorm/subSubJmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0;model_1/basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_2/batchnorm/subк
7model_1/basemodel/batch_normalization_2/batchnorm/add_1AddV2;model_1/basemodel/batch_normalization_2/batchnorm/mul_1:z:09model_1/basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@29
7model_1/basemodel/batch_normalization_2/batchnorm/add_1К
@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpImodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02B
@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp╖
7model_1/basemodel/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:29
7model_1/basemodel/batch_normalization_1/batchnorm/add/yи
5model_1/basemodel/batch_normalization_1/batchnorm/addAddV2Hmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:0@model_1/basemodel/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_1/batchnorm/add█
7model_1/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt9model_1/basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@29
7model_1/basemodel/batch_normalization_1/batchnorm/RsqrtЦ
Dmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpMmodel_1_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpе
5model_1/basemodel/batch_normalization_1/batchnorm/mulMul;model_1/basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Lmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_1/batchnorm/mulЯ
7model_1/basemodel/batch_normalization_1/batchnorm/mul_1Mul2model_1/basemodel/stream_1_conv_1/BiasAdd:output:09model_1/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@29
7model_1/basemodel/batch_normalization_1/batchnorm/mul_1Р
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpKmodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1е
7model_1/basemodel/batch_normalization_1/batchnorm/mul_2MulJmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1:value:09model_1/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@29
7model_1/basemodel/batch_normalization_1/batchnorm/mul_2Р
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpKmodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2г
5model_1/basemodel/batch_normalization_1/batchnorm/subSubJmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0;model_1/basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_1/batchnorm/subк
7model_1/basemodel/batch_normalization_1/batchnorm/add_1AddV2;model_1/basemodel/batch_normalization_1/batchnorm/mul_1:z:09model_1/basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@29
7model_1/basemodel/batch_normalization_1/batchnorm/add_1Д
>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOpGmodel_1_basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp│
5model_1/basemodel/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:27
5model_1/basemodel/batch_normalization/batchnorm/add/yа
3model_1/basemodel/batch_normalization/batchnorm/addAddV2Fmodel_1/basemodel/batch_normalization/batchnorm/ReadVariableOp:value:0>model_1/basemodel/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@25
3model_1/basemodel/batch_normalization/batchnorm/add╒
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
3model_1/basemodel/batch_normalization/batchnorm/mulЩ
5model_1/basemodel/batch_normalization/batchnorm/mul_1Mul2model_1/basemodel/stream_0_conv_1/BiasAdd:output:07model_1/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ▄@27
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
3model_1/basemodel/batch_normalization/batchnorm/subв
5model_1/basemodel/batch_normalization/batchnorm/add_1AddV29model_1/basemodel/batch_normalization/batchnorm/mul_1:z:07model_1/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ▄@27
5model_1/basemodel/batch_normalization/batchnorm/add_1╞
#model_1/basemodel/activation_2/ReluRelu;model_1/basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ▄@2%
#model_1/basemodel/activation_2/Relu╞
#model_1/basemodel/activation_1/ReluRelu;model_1/basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ▄@2%
#model_1/basemodel/activation_1/Relu└
!model_1/basemodel/activation/ReluRelu9model_1/basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ▄@2#
!model_1/basemodel/activation/Relu╬
*model_1/basemodel/stream_2_drop_1/IdentityIdentity1model_1/basemodel/activation_2/Relu:activations:0*
T0*,
_output_shapes
:         ▄@2,
*model_1/basemodel/stream_2_drop_1/Identity╬
*model_1/basemodel/stream_1_drop_1/IdentityIdentity1model_1/basemodel/activation_1/Relu:activations:0*
T0*,
_output_shapes
:         ▄@2,
*model_1/basemodel/stream_1_drop_1/Identity╠
*model_1/basemodel/stream_0_drop_1/IdentityIdentity/model_1/basemodel/activation/Relu:activations:0*
T0*,
_output_shapes
:         ▄@2,
*model_1/basemodel/stream_0_drop_1/Identity╚
Amodel_1/basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Amodel_1/basemodel/global_average_pooling1d/Mean/reduction_indicesЭ
/model_1/basemodel/global_average_pooling1d/MeanMean3model_1/basemodel/stream_0_drop_1/Identity:output:0Jmodel_1/basemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @21
/model_1/basemodel/global_average_pooling1d/Mean╠
Cmodel_1/basemodel/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2E
Cmodel_1/basemodel/global_average_pooling1d_1/Mean/reduction_indicesг
1model_1/basemodel/global_average_pooling1d_1/MeanMean3model_1/basemodel/stream_1_drop_1/Identity:output:0Lmodel_1/basemodel/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @23
1model_1/basemodel/global_average_pooling1d_1/Mean╠
Cmodel_1/basemodel/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2E
Cmodel_1/basemodel/global_average_pooling1d_2/Mean/reduction_indicesг
1model_1/basemodel/global_average_pooling1d_2/MeanMean3model_1/basemodel/stream_2_drop_1/Identity:output:0Lmodel_1/basemodel/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @23
1model_1/basemodel/global_average_pooling1d_2/MeanШ
)model_1/basemodel/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_1/basemodel/concatenate/concat/axis·
$model_1/basemodel/concatenate/concatConcatV28model_1/basemodel/global_average_pooling1d/Mean:output:0:model_1/basemodel/global_average_pooling1d_1/Mean:output:0:model_1/basemodel/global_average_pooling1d_2/Mean:output:02model_1/basemodel/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:         └2&
$model_1/basemodel/concatenate/concat╞
*model_1/basemodel/dense_1_dropout/IdentityIdentity-model_1/basemodel/concatenate/concat:output:0*
T0*(
_output_shapes
:         └2,
*model_1/basemodel/dense_1_dropout/Identity▄
/model_1/basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp8model_1_basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	└T*
dtype021
/model_1/basemodel/dense_1/MatMul/ReadVariableOpю
 model_1/basemodel/dense_1/MatMulMatMul3model_1/basemodel/dense_1_dropout/Identity:output:07model_1/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2"
 model_1/basemodel/dense_1/MatMul┌
0model_1/basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp9model_1_basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype022
0model_1/basemodel/dense_1/BiasAdd/ReadVariableOpщ
!model_1/basemodel/dense_1/BiasAddBiasAdd*model_1/basemodel/dense_1/MatMul:product:08model_1/basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2#
!model_1/basemodel/dense_1/BiasAddК
@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpImodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02B
@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp╖
7model_1/basemodel/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:29
7model_1/basemodel/batch_normalization_3/batchnorm/add/yи
5model_1/basemodel/batch_normalization_3/batchnorm/addAddV2Hmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp:value:0@model_1/basemodel/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:T27
5model_1/basemodel/batch_normalization_3/batchnorm/add█
7model_1/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt9model_1/basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:T29
7model_1/basemodel/batch_normalization_3/batchnorm/RsqrtЦ
Dmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpMmodel_1_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02F
Dmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpе
5model_1/basemodel/batch_normalization_3/batchnorm/mulMul;model_1/basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Lmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T27
5model_1/basemodel/batch_normalization_3/batchnorm/mulТ
7model_1/basemodel/batch_normalization_3/batchnorm/mul_1Mul*model_1/basemodel/dense_1/BiasAdd:output:09model_1/basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T29
7model_1/basemodel/batch_normalization_3/batchnorm/mul_1Р
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpKmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02D
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1е
7model_1/basemodel/batch_normalization_3/batchnorm/mul_2MulJmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:value:09model_1/basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:T29
7model_1/basemodel/batch_normalization_3/batchnorm/mul_2Р
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpKmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02D
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2г
5model_1/basemodel/batch_normalization_3/batchnorm/subSubJmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2:value:0;model_1/basemodel/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T27
5model_1/basemodel/batch_normalization_3/batchnorm/subе
7model_1/basemodel/batch_normalization_3/batchnorm/add_1AddV2;model_1/basemodel/batch_normalization_3/batchnorm/mul_1:z:09model_1/basemodel/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T29
7model_1/basemodel/batch_normalization_3/batchnorm/add_1╓
,model_1/basemodel/dense_activation_1/SigmoidSigmoid;model_1/basemodel/batch_normalization_3/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2.
,model_1/basemodel/dense_activation_1/SigmoidЛ
IdentityIdentity0model_1/basemodel/dense_activation_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         T2

IdentityБ
NoOpNoOp?^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOpA^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1A^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2C^model_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp1^model_1/basemodel/dense_1/BiasAdd/ReadVariableOp0^model_1/basemodel/dense_1/MatMul/ReadVariableOp9^model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp9^model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp9^model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 2А
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
8model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp8model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp2М
Dmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpDmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp2t
8model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp8model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp2М
Dmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpDmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:Y U
,
_output_shapes
:         ▄
%
_user_specified_nameleft_inputs
Д
╫
M__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_13978824

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ▄2
conv1d/ExpandDims╕
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
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ▄@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ▄@*
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▄@2	
BiasAddф
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareй
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const┌
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identity╟
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▄: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
є
╙
8__inference_batch_normalization_2_layer_call_fn_13982213

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_139788762
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▄@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
С	
╙
8__inference_batch_normalization_2_layer_call_fn_13982200

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallл
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
GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_139784322
StatefulPartitionedCallИ
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
°=
╡	
E__inference_model_1_layer_call_and_return_conditional_losses_13980471
left_inputs(
basemodel_13980397:@ 
basemodel_13980399:@(
basemodel_13980401:@ 
basemodel_13980403:@(
basemodel_13980405:@ 
basemodel_13980407:@ 
basemodel_13980409:@ 
basemodel_13980411:@ 
basemodel_13980413:@ 
basemodel_13980415:@ 
basemodel_13980417:@ 
basemodel_13980419:@ 
basemodel_13980421:@ 
basemodel_13980423:@ 
basemodel_13980425:@ 
basemodel_13980427:@ 
basemodel_13980429:@ 
basemodel_13980431:@%
basemodel_13980433:	└T 
basemodel_13980435:T 
basemodel_13980437:T 
basemodel_13980439:T 
basemodel_13980441:T 
basemodel_13980443:T
identityИв!basemodel/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpз
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsleft_inputsleft_inputsbasemodel_13980397basemodel_13980399basemodel_13980401basemodel_13980403basemodel_13980405basemodel_13980407basemodel_13980409basemodel_13980411basemodel_13980413basemodel_13980415basemodel_13980417basemodel_13980419basemodel_13980421basemodel_13980423basemodel_13980425basemodel_13980427basemodel_13980429basemodel_13980431basemodel_13980433basemodel_13980435basemodel_13980437basemodel_13980439basemodel_13980441basemodel_13980443*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_basemodel_layer_call_and_return_conditional_losses_139790872#
!basemodel/StatefulPartitionedCall┼
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_13980405*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absй
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul╦
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_13980401*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareй
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const┌
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┼
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_13980397*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absй
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul▓
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_13980433*
_output_shapes
:	└T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└T2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const╖
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЕ
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

Identity═
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:Y U
,
_output_shapes
:         ▄
%
_user_specified_nameleft_inputs
√
p
Q__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_13981692

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ▄2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ▄*
dtype0*
seed╖*
seed2╖2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ▄2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
ў
l
M__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_13982418

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ▄@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ▄@*
dtype0*
seed╖*
seed2╖2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ▄@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ▄@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ▄@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ▄@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
╫=
░	
E__inference_model_1_layer_call_and_return_conditional_losses_13980290

inputs(
basemodel_13980216:@ 
basemodel_13980218:@(
basemodel_13980220:@ 
basemodel_13980222:@(
basemodel_13980224:@ 
basemodel_13980226:@ 
basemodel_13980228:@ 
basemodel_13980230:@ 
basemodel_13980232:@ 
basemodel_13980234:@ 
basemodel_13980236:@ 
basemodel_13980238:@ 
basemodel_13980240:@ 
basemodel_13980242:@ 
basemodel_13980244:@ 
basemodel_13980246:@ 
basemodel_13980248:@ 
basemodel_13980250:@%
basemodel_13980252:	└T 
basemodel_13980254:T 
basemodel_13980256:T 
basemodel_13980258:T 
basemodel_13980260:T 
basemodel_13980262:T
identityИв!basemodel/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpР
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsbasemodel_13980216basemodel_13980218basemodel_13980220basemodel_13980222basemodel_13980224basemodel_13980226basemodel_13980228basemodel_13980230basemodel_13980232basemodel_13980234basemodel_13980236basemodel_13980238basemodel_13980240basemodel_13980242basemodel_13980244basemodel_13980246basemodel_13980248basemodel_13980250basemodel_13980252basemodel_13980254basemodel_13980256basemodel_13980258basemodel_13980260basemodel_13980262*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_basemodel_layer_call_and_return_conditional_losses_139797182#
!basemodel/StatefulPartitionedCall┼
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_13980224*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absй
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul╦
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_13980220*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareй
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const┌
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┼
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_13980216*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absй
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul▓
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_13980252*
_output_shapes
:	└T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└T2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const╖
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЕ
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

Identity═
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
√i
╝
$__inference__traced_restore_13982887
file_prefix=
'assignvariableop_stream_0_conv_1_kernel:@5
'assignvariableop_1_stream_0_conv_1_bias:@?
)assignvariableop_2_stream_1_conv_1_kernel:@5
'assignvariableop_3_stream_1_conv_1_bias:@?
)assignvariableop_4_stream_2_conv_1_kernel:@5
'assignvariableop_5_stream_2_conv_1_bias:@:
,assignvariableop_6_batch_normalization_gamma:@9
+assignvariableop_7_batch_normalization_beta:@<
.assignvariableop_8_batch_normalization_1_gamma:@;
-assignvariableop_9_batch_normalization_1_beta:@=
/assignvariableop_10_batch_normalization_2_gamma:@<
.assignvariableop_11_batch_normalization_2_beta:@5
"assignvariableop_12_dense_1_kernel:	└T.
 assignvariableop_13_dense_1_bias:T=
/assignvariableop_14_batch_normalization_3_gamma:T<
.assignvariableop_15_batch_normalization_3_beta:TA
3assignvariableop_16_batch_normalization_moving_mean:@E
7assignvariableop_17_batch_normalization_moving_variance:@C
5assignvariableop_18_batch_normalization_1_moving_mean:@G
9assignvariableop_19_batch_normalization_1_moving_variance:@C
5assignvariableop_20_batch_normalization_2_moving_mean:@G
9assignvariableop_21_batch_normalization_2_moving_variance:@C
5assignvariableop_22_batch_normalization_3_moving_mean:TG
9assignvariableop_23_batch_normalization_3_moving_variance:T
identity_25ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9У

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Я	
valueХ	BТ	B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names└
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesи
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

Identityж
AssignVariableOpAssignVariableOp'assignvariableop_stream_0_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1м
AssignVariableOp_1AssignVariableOp'assignvariableop_1_stream_0_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2о
AssignVariableOp_2AssignVariableOp)assignvariableop_2_stream_1_conv_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3м
AssignVariableOp_3AssignVariableOp'assignvariableop_3_stream_1_conv_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4о
AssignVariableOp_4AssignVariableOp)assignvariableop_4_stream_2_conv_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5м
AssignVariableOp_5AssignVariableOp'assignvariableop_5_stream_2_conv_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6▒
AssignVariableOp_6AssignVariableOp,assignvariableop_6_batch_normalization_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7░
AssignVariableOp_7AssignVariableOp+assignvariableop_7_batch_normalization_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8│
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9▓
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10╖
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_2_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11╢
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_2_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12к
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13и
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14╖
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_3_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15╢
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_3_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16╗
AssignVariableOp_16AssignVariableOp3assignvariableop_16_batch_normalization_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17┐
AssignVariableOp_17AssignVariableOp7assignvariableop_17_batch_normalization_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18╜
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_1_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19┴
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_1_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20╜
AssignVariableOp_20AssignVariableOp5assignvariableop_20_batch_normalization_2_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21┴
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_normalization_2_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22╜
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23┴
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpю
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24f
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_25╓
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
▀=
░	
E__inference_model_1_layer_call_and_return_conditional_losses_13980107

inputs(
basemodel_13980033:@ 
basemodel_13980035:@(
basemodel_13980037:@ 
basemodel_13980039:@(
basemodel_13980041:@ 
basemodel_13980043:@ 
basemodel_13980045:@ 
basemodel_13980047:@ 
basemodel_13980049:@ 
basemodel_13980051:@ 
basemodel_13980053:@ 
basemodel_13980055:@ 
basemodel_13980057:@ 
basemodel_13980059:@ 
basemodel_13980061:@ 
basemodel_13980063:@ 
basemodel_13980065:@ 
basemodel_13980067:@%
basemodel_13980069:	└T 
basemodel_13980071:T 
basemodel_13980073:T 
basemodel_13980075:T 
basemodel_13980077:T 
basemodel_13980079:T
identityИв!basemodel/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpШ
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsbasemodel_13980033basemodel_13980035basemodel_13980037basemodel_13980039basemodel_13980041basemodel_13980043basemodel_13980045basemodel_13980047basemodel_13980049basemodel_13980051basemodel_13980053basemodel_13980055basemodel_13980057basemodel_13980059basemodel_13980061basemodel_13980063basemodel_13980065basemodel_13980067basemodel_13980069basemodel_13980071basemodel_13980073basemodel_13980075basemodel_13980077basemodel_13980079*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_basemodel_layer_call_and_return_conditional_losses_139790872#
!basemodel/StatefulPartitionedCall┼
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_13980041*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absй
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul╦
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_13980037*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareй
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_1_conv_1/kernel/Regularizer/Const┌
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┼
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_13980033*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absй
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_2_conv_1/kernel/Regularizer/Const╫
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul▓
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_13980069*
_output_shapes
:	└T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└T2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const╖
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЕ
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         T2

Identity═
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:         ▄
 
_user_specified_nameinputs
╣
▓
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_13982086

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
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
 :                  @2
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
└
Г
,__inference_basemodel_layer_call_fn_13981209
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

unknown_17:	└T

unknown_18:T

unknown_19:T

unknown_20:T

unknown_21:T

unknown_22:T
identityИвStatefulPartitionedCall╜
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
:         T*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_basemodel_layer_call_and_return_conditional_losses_139790872
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
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         ▄:         ▄:         ▄: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:         ▄
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:         ▄
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:         ▄
"
_user_specified_name
inputs/2
■
k
M__inference_dense_1_dropout_layer_call_and_return_conditional_losses_13979022

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         └2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         └2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         └:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
К
t
X__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_13979005

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
:         @2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
╚
k
2__inference_stream_1_drop_1_layer_call_fn_13982401

inputs
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ▄@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_139792352
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ▄@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▄@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ▄@
 
_user_specified_nameinputs
ї
з
E__inference_dense_1_layer_call_and_return_conditional_losses_13979040

inputs1
matmul_readvariableop_resource:	└T-
biasadd_readvariableop_resource:T
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└T*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2	
BiasAdd╛
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└T2 
dense_1/kernel/Regularizer/AbsХ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const╖
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
╫#<2"
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

Identityп
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs"иL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╣
serving_defaultе
H
left_inputs9
serving_default_left_inputs:0         ▄=
	basemodel0
StatefulPartitionedCall:0         Ttensorflow/serving/predict:╫╬
Л
layer-0
layer_with_weights-0
layer-1
regularization_losses
trainable_variables
	variables
	keras_api

signatures
Ы__call__
+Ь&call_and_return_all_conditional_losses
Э_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
█
layer-0
	layer-1

layer-2
layer-3
layer-4
layer-5
layer_with_weights-0
layer-6
layer_with_weights-1
layer-7
layer_with_weights-2
layer-8
layer_with_weights-3
layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer_with_weights-6
layer-23
 layer_with_weights-7
 layer-24
!layer-25
"regularization_losses
#trainable_variables
$	variables
%	keras_api
Ю__call__
+Я&call_and_return_all_conditional_losses"
_tf_keras_network
 "
trackable_list_wrapper
Ц
&0
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111
212
313
414
515"
trackable_list_wrapper
╓
&0
'1
(2
)3
*4
+5
,6
-7
68
79
.10
/11
812
913
014
115
:16
;17
218
319
420
521
<22
=23"
trackable_list_wrapper
╬
>layer_metrics
regularization_losses
?layer_regularization_losses
trainable_variables
	variables
@non_trainable_variables

Alayers
Bmetrics
Ы__call__
Э_default_save_signature
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
-
аserving_default"
signature_map
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
з
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
б__call__
+в&call_and_return_all_conditional_losses"
_tf_keras_layer
з
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
г__call__
+д&call_and_return_all_conditional_losses"
_tf_keras_layer
з
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
е__call__
+ж&call_and_return_all_conditional_losses"
_tf_keras_layer
╜

&kernel
'bias
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
з__call__
+и&call_and_return_all_conditional_losses"
_tf_keras_layer
╜

(kernel
)bias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
й__call__
+к&call_and_return_all_conditional_losses"
_tf_keras_layer
╜

*kernel
+bias
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
л__call__
+м&call_and_return_all_conditional_losses"
_tf_keras_layer
ь
[axis
	,gamma
-beta
6moving_mean
7moving_variance
\regularization_losses
]trainable_variables
^	variables
_	keras_api
н__call__
+о&call_and_return_all_conditional_losses"
_tf_keras_layer
ь
`axis
	.gamma
/beta
8moving_mean
9moving_variance
aregularization_losses
btrainable_variables
c	variables
d	keras_api
п__call__
+░&call_and_return_all_conditional_losses"
_tf_keras_layer
ь
eaxis
	0gamma
1beta
:moving_mean
;moving_variance
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses"
_tf_keras_layer
з
jregularization_losses
ktrainable_variables
l	variables
m	keras_api
│__call__
+┤&call_and_return_all_conditional_losses"
_tf_keras_layer
з
nregularization_losses
otrainable_variables
p	variables
q	keras_api
╡__call__
+╢&call_and_return_all_conditional_losses"
_tf_keras_layer
з
rregularization_losses
strainable_variables
t	variables
u	keras_api
╖__call__
+╕&call_and_return_all_conditional_losses"
_tf_keras_layer
з
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
╣__call__
+║&call_and_return_all_conditional_losses"
_tf_keras_layer
з
zregularization_losses
{trainable_variables
|	variables
}	keras_api
╗__call__
+╝&call_and_return_all_conditional_losses"
_tf_keras_layer
й
~regularization_losses
trainable_variables
А	variables
Б	keras_api
╜__call__
+╛&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Вregularization_losses
Гtrainable_variables
Д	variables
Е	keras_api
┐__call__
+└&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Жregularization_losses
Зtrainable_variables
И	variables
Й	keras_api
┴__call__
+┬&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Кregularization_losses
Лtrainable_variables
М	variables
Н	keras_api
├__call__
+─&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Оregularization_losses
Пtrainable_variables
Р	variables
С	keras_api
┼__call__
+╞&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Тregularization_losses
Уtrainable_variables
Ф	variables
Х	keras_api
╟__call__
+╚&call_and_return_all_conditional_losses"
_tf_keras_layer
┴

2kernel
3bias
Цregularization_losses
Чtrainable_variables
Ш	variables
Щ	keras_api
╔__call__
+╩&call_and_return_all_conditional_losses"
_tf_keras_layer
ё
	Ъaxis
	4gamma
5beta
<moving_mean
=moving_variance
Ыregularization_losses
Ьtrainable_variables
Э	variables
Ю	keras_api
╦__call__
+╠&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Яregularization_losses
аtrainable_variables
б	variables
в	keras_api
═__call__
+╬&call_and_return_all_conditional_losses"
_tf_keras_layer
@
╧0
╨1
╤2
╥3"
trackable_list_wrapper
Ц
&0
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111
212
313
414
515"
trackable_list_wrapper
╓
&0
'1
(2
)3
*4
+5
,6
-7
68
79
.10
/11
812
913
014
115
:16
;17
218
319
420
521
<22
=23"
trackable_list_wrapper
╡
гlayer_metrics
"regularization_losses
 дlayer_regularization_losses
#trainable_variables
$	variables
еnon_trainable_variables
жlayers
зmetrics
Ю__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
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
!:	└T2dense_1/kernel
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
60
71
82
93
:4
;5
<6
=7"
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
╡
иmetrics
йlayer_metrics
Cregularization_losses
Dtrainable_variables
E	variables
кnon_trainable_variables
лlayers
 мlayer_regularization_losses
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
нmetrics
оlayer_metrics
Gregularization_losses
Htrainable_variables
I	variables
пnon_trainable_variables
░layers
 ▒layer_regularization_losses
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
▓metrics
│layer_metrics
Kregularization_losses
Ltrainable_variables
M	variables
┤non_trainable_variables
╡layers
 ╢layer_regularization_losses
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
(
╧0"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
╡
╖metrics
╕layer_metrics
Oregularization_losses
Ptrainable_variables
Q	variables
╣non_trainable_variables
║layers
 ╗layer_regularization_losses
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
(
╨0"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
╡
╝metrics
╜layer_metrics
Sregularization_losses
Ttrainable_variables
U	variables
╛non_trainable_variables
┐layers
 └layer_regularization_losses
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
(
╤0"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
╡
┴metrics
┬layer_metrics
Wregularization_losses
Xtrainable_variables
Y	variables
├non_trainable_variables
─layers
 ┼layer_regularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
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
╡
╞metrics
╟layer_metrics
\regularization_losses
]trainable_variables
^	variables
╚non_trainable_variables
╔layers
 ╩layer_regularization_losses
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
<
.0
/1
82
93"
trackable_list_wrapper
╡
╦metrics
╠layer_metrics
aregularization_losses
btrainable_variables
c	variables
═non_trainable_variables
╬layers
 ╧layer_regularization_losses
п__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
<
00
11
:2
;3"
trackable_list_wrapper
╡
╨metrics
╤layer_metrics
fregularization_losses
gtrainable_variables
h	variables
╥non_trainable_variables
╙layers
 ╘layer_regularization_losses
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
╡
╒metrics
╓layer_metrics
jregularization_losses
ktrainable_variables
l	variables
╫non_trainable_variables
╪layers
 ┘layer_regularization_losses
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
╡
┌metrics
█layer_metrics
nregularization_losses
otrainable_variables
p	variables
▄non_trainable_variables
▌layers
 ▐layer_regularization_losses
╡__call__
+╢&call_and_return_all_conditional_losses
'╢"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
▀metrics
рlayer_metrics
rregularization_losses
strainable_variables
t	variables
сnon_trainable_variables
тlayers
 уlayer_regularization_losses
╖__call__
+╕&call_and_return_all_conditional_losses
'╕"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
фmetrics
хlayer_metrics
vregularization_losses
wtrainable_variables
x	variables
цnon_trainable_variables
чlayers
 шlayer_regularization_losses
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
╡
щmetrics
ъlayer_metrics
zregularization_losses
{trainable_variables
|	variables
ыnon_trainable_variables
ьlayers
 эlayer_regularization_losses
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
╢
юmetrics
яlayer_metrics
~regularization_losses
trainable_variables
А	variables
Ёnon_trainable_variables
ёlayers
 Єlayer_regularization_losses
╜__call__
+╛&call_and_return_all_conditional_losses
'╛"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
єmetrics
Їlayer_metrics
Вregularization_losses
Гtrainable_variables
Д	variables
їnon_trainable_variables
Ўlayers
 ўlayer_regularization_losses
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
╕
°metrics
∙layer_metrics
Жregularization_losses
Зtrainable_variables
И	variables
·non_trainable_variables
√layers
 №layer_regularization_losses
┴__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
¤metrics
■layer_metrics
Кregularization_losses
Лtrainable_variables
М	variables
 non_trainable_variables
Аlayers
 Бlayer_regularization_losses
├__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Вmetrics
Гlayer_metrics
Оregularization_losses
Пtrainable_variables
Р	variables
Дnon_trainable_variables
Еlayers
 Жlayer_regularization_losses
┼__call__
+╞&call_and_return_all_conditional_losses
'╞"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Зmetrics
Иlayer_metrics
Тregularization_losses
Уtrainable_variables
Ф	variables
Йnon_trainable_variables
Кlayers
 Лlayer_regularization_losses
╟__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
(
╥0"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
╕
Мmetrics
Нlayer_metrics
Цregularization_losses
Чtrainable_variables
Ш	variables
Оnon_trainable_variables
Пlayers
 Рlayer_regularization_losses
╔__call__
+╩&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
<
40
51
<2
=3"
trackable_list_wrapper
╕
Сmetrics
Тlayer_metrics
Ыregularization_losses
Ьtrainable_variables
Э	variables
Уnon_trainable_variables
Фlayers
 Хlayer_regularization_losses
╦__call__
+╠&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Цmetrics
Чlayer_metrics
Яregularization_losses
аtrainable_variables
б	variables
Шnon_trainable_variables
Щlayers
 Ъlayer_regularization_losses
═__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
X
60
71
82
93
:4
;5
<6
=7"
trackable_list_wrapper
ц
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
22
23
 24
!25"
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
╧0"
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
╨0"
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
╤0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
60
71"
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
80
91"
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
:0
;1"
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
╥0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
<0
=1"
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
Ў2є
*__inference_model_1_layer_call_fn_13980158
*__inference_model_1_layer_call_fn_13980680
*__inference_model_1_layer_call_fn_13980733
*__inference_model_1_layer_call_fn_13980394└
╖▓│
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
kwonlydefaultsк 
annotationsк *
 
т2▀
E__inference_model_1_layer_call_and_return_conditional_losses_13980883
E__inference_model_1_layer_call_and_return_conditional_losses_13981130
E__inference_model_1_layer_call_and_return_conditional_losses_13980471
E__inference_model_1_layer_call_and_return_conditional_losses_13980548└
╖▓│
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
kwonlydefaultsк 
annotationsк *
 
╥B╧
#__inference__wrapped_model_13978024left_inputs"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■2√
,__inference_basemodel_layer_call_fn_13979138
,__inference_basemodel_layer_call_fn_13981209
,__inference_basemodel_layer_call_fn_13981264
,__inference_basemodel_layer_call_fn_13979824└
╖▓│
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
kwonlydefaultsк 
annotationsк *
 
ъ2ч
G__inference_basemodel_layer_call_and_return_conditional_losses_13981416
G__inference_basemodel_layer_call_and_return_conditional_losses_13981665
G__inference_basemodel_layer_call_and_return_conditional_losses_13979925
G__inference_basemodel_layer_call_and_return_conditional_losses_13980026└
╖▓│
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
kwonlydefaultsк 
annotationsк *
 
╤B╬
&__inference_signature_wrapper_13980627left_inputs"Ф
Н▓Й
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
annotationsк *
 
к2з
6__inference_stream_0_input_drop_layer_call_fn_13981670
6__inference_stream_0_input_drop_layer_call_fn_13981675┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
р2▌
Q__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_13981680
Q__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_13981692┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
к2з
6__inference_stream_1_input_drop_layer_call_fn_13981697
6__inference_stream_1_input_drop_layer_call_fn_13981702┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
р2▌
Q__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_13981707
Q__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_13981719┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
к2з
6__inference_stream_2_input_drop_layer_call_fn_13981724
6__inference_stream_2_input_drop_layer_call_fn_13981729┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
р2▌
Q__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_13981734
Q__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_13981746┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
▄2┘
2__inference_stream_0_conv_1_layer_call_fn_13981761в
Щ▓Х
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
annotationsк *
 
ў2Ї
M__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_13981782в
Щ▓Х
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
annotationsк *
 
▄2┘
2__inference_stream_1_conv_1_layer_call_fn_13981797в
Щ▓Х
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
annotationsк *
 
ў2Ї
M__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_13981818в
Щ▓Х
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
annotationsк *
 
▄2┘
2__inference_stream_2_conv_1_layer_call_fn_13981833в
Щ▓Х
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
annotationsк *
 
ў2Ї
M__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_13981854в
Щ▓Х
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
annotationsк *
 
Ъ2Ч
6__inference_batch_normalization_layer_call_fn_13981867
6__inference_batch_normalization_layer_call_fn_13981880
6__inference_batch_normalization_layer_call_fn_13981893
6__inference_batch_normalization_layer_call_fn_13981906┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
Ж2Г
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_13981926
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_13981960
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_13981980
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_13982014┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
в2Я
8__inference_batch_normalization_1_layer_call_fn_13982027
8__inference_batch_normalization_1_layer_call_fn_13982040
8__inference_batch_normalization_1_layer_call_fn_13982053
8__inference_batch_normalization_1_layer_call_fn_13982066┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
О2Л
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_13982086
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_13982120
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_13982140
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_13982174┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
в2Я
8__inference_batch_normalization_2_layer_call_fn_13982187
8__inference_batch_normalization_2_layer_call_fn_13982200
8__inference_batch_normalization_2_layer_call_fn_13982213
8__inference_batch_normalization_2_layer_call_fn_13982226┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
О2Л
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13982246
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13982280
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13982300
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13982334┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
╫2╘
-__inference_activation_layer_call_fn_13982339в
Щ▓Х
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
annotationsк *
 
Є2я
H__inference_activation_layer_call_and_return_conditional_losses_13982344в
Щ▓Х
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
annotationsк *
 
┘2╓
/__inference_activation_1_layer_call_fn_13982349в
Щ▓Х
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
annotationsк *
 
Ї2ё
J__inference_activation_1_layer_call_and_return_conditional_losses_13982354в
Щ▓Х
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
annotationsк *
 
┘2╓
/__inference_activation_2_layer_call_fn_13982359в
Щ▓Х
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
annotationsк *
 
Ї2ё
J__inference_activation_2_layer_call_and_return_conditional_losses_13982364в
Щ▓Х
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
annotationsк *
 
в2Я
2__inference_stream_0_drop_1_layer_call_fn_13982369
2__inference_stream_0_drop_1_layer_call_fn_13982374┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
╪2╒
M__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_13982379
M__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_13982391┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
в2Я
2__inference_stream_1_drop_1_layer_call_fn_13982396
2__inference_stream_1_drop_1_layer_call_fn_13982401┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
╪2╒
M__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_13982406
M__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_13982418┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
в2Я
2__inference_stream_2_drop_1_layer_call_fn_13982423
2__inference_stream_2_drop_1_layer_call_fn_13982428┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
╪2╒
M__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_13982433
M__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_13982445┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
п2м
;__inference_global_average_pooling1d_layer_call_fn_13982450
;__inference_global_average_pooling1d_layer_call_fn_13982455п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
х2т
V__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_13982461
V__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_13982467п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
│2░
=__inference_global_average_pooling1d_1_layer_call_fn_13982472
=__inference_global_average_pooling1d_1_layer_call_fn_13982477п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
щ2ц
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_13982483
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_13982489п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
│2░
=__inference_global_average_pooling1d_2_layer_call_fn_13982494
=__inference_global_average_pooling1d_2_layer_call_fn_13982499п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
щ2ц
X__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_13982505
X__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_13982511п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╪2╒
.__inference_concatenate_layer_call_fn_13982518в
Щ▓Х
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
annotationsк *
 
є2Ё
I__inference_concatenate_layer_call_and_return_conditional_losses_13982526в
Щ▓Х
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
annotationsк *
 
в2Я
2__inference_dense_1_dropout_layer_call_fn_13982531
2__inference_dense_1_dropout_layer_call_fn_13982536┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
╪2╒
M__inference_dense_1_dropout_layer_call_and_return_conditional_losses_13982541
M__inference_dense_1_dropout_layer_call_and_return_conditional_losses_13982545┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
╘2╤
*__inference_dense_1_layer_call_fn_13982560в
Щ▓Х
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
annotationsк *
 
я2ь
E__inference_dense_1_layer_call_and_return_conditional_losses_13982576в
Щ▓Х
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
annotationsк *
 
о2л
8__inference_batch_normalization_3_layer_call_fn_13982589
8__inference_batch_normalization_3_layer_call_fn_13982602┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
ф2с
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_13982622
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_13982656┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
▀2▄
5__inference_dense_activation_1_layer_call_fn_13982661в
Щ▓Х
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
annotationsк *
 
·2ў
P__inference_dense_activation_1_layer_call_and_return_conditional_losses_13982666в
Щ▓Х
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
annotationsк *
 
╡2▓
__inference_loss_fn_0_13982677П
З▓Г
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
annotationsк *в 
╡2▓
__inference_loss_fn_1_13982688П
З▓Г
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
annotationsк *в 
╡2▓
__inference_loss_fn_2_13982699П
З▓Г
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
annotationsк *в 
╡2▓
__inference_loss_fn_3_13982710П
З▓Г
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
annotationsк *в ┤
#__inference__wrapped_model_13978024М*+()&';0:19.8/7,6-23=4<59в6
/в,
*К'
left_inputs         ▄
к "5к2
0
	basemodel#К 
	basemodel         T░
J__inference_activation_1_layer_call_and_return_conditional_losses_13982354b4в1
*в'
%К"
inputs         ▄@
к "*в'
 К
0         ▄@
Ъ И
/__inference_activation_1_layer_call_fn_13982349U4в1
*в'
%К"
inputs         ▄@
к "К         ▄@░
J__inference_activation_2_layer_call_and_return_conditional_losses_13982364b4в1
*в'
%К"
inputs         ▄@
к "*в'
 К
0         ▄@
Ъ И
/__inference_activation_2_layer_call_fn_13982359U4в1
*в'
%К"
inputs         ▄@
к "К         ▄@о
H__inference_activation_layer_call_and_return_conditional_losses_13982344b4в1
*в'
%К"
inputs         ▄@
к "*в'
 К
0         ▄@
Ъ Ж
-__inference_activation_layer_call_fn_13982339U4в1
*в'
%К"
inputs         ▄@
к "К         ▄@и
G__inference_basemodel_layer_call_and_return_conditional_losses_13979925▄*+()&';0:19.8/7,6-23=4<5ШвФ
МвИ
~Ъ{
'К$
inputs_0         ▄
'К$
inputs_1         ▄
'К$
inputs_2         ▄
p 

 
к "%в"
К
0         T
Ъ и
G__inference_basemodel_layer_call_and_return_conditional_losses_13980026▄*+()&':;0189./67,-23<=45ШвФ
МвИ
~Ъ{
'К$
inputs_0         ▄
'К$
inputs_1         ▄
'К$
inputs_2         ▄
p

 
к "%в"
К
0         T
Ъ и
G__inference_basemodel_layer_call_and_return_conditional_losses_13981416▄*+()&';0:19.8/7,6-23=4<5ШвФ
МвИ
~Ъ{
'К$
inputs/0         ▄
'К$
inputs/1         ▄
'К$
inputs/2         ▄
p 

 
к "%в"
К
0         T
Ъ и
G__inference_basemodel_layer_call_and_return_conditional_losses_13981665▄*+()&':;0189./67,-23<=45ШвФ
МвИ
~Ъ{
'К$
inputs/0         ▄
'К$
inputs/1         ▄
'К$
inputs/2         ▄
p

 
к "%в"
К
0         T
Ъ А
,__inference_basemodel_layer_call_fn_13979138╧*+()&';0:19.8/7,6-23=4<5ШвФ
МвИ
~Ъ{
'К$
inputs_0         ▄
'К$
inputs_1         ▄
'К$
inputs_2         ▄
p 

 
к "К         TА
,__inference_basemodel_layer_call_fn_13979824╧*+()&':;0189./67,-23<=45ШвФ
МвИ
~Ъ{
'К$
inputs_0         ▄
'К$
inputs_1         ▄
'К$
inputs_2         ▄
p

 
к "К         TА
,__inference_basemodel_layer_call_fn_13981209╧*+()&';0:19.8/7,6-23=4<5ШвФ
МвИ
~Ъ{
'К$
inputs/0         ▄
'К$
inputs/1         ▄
'К$
inputs/2         ▄
p 

 
к "К         TА
,__inference_basemodel_layer_call_fn_13981264╧*+()&':;0189./67,-23<=45ШвФ
МвИ
~Ъ{
'К$
inputs/0         ▄
'К$
inputs/1         ▄
'К$
inputs/2         ▄
p

 
к "К         T╙
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_13982086|9.8/@в=
6в3
-К*
inputs                  @
p 
к "2в/
(К%
0                  @
Ъ ╙
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_13982120|89./@в=
6в3
-К*
inputs                  @
p
к "2в/
(К%
0                  @
Ъ ├
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_13982140l9.8/8в5
.в+
%К"
inputs         ▄@
p 
к "*в'
 К
0         ▄@
Ъ ├
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_13982174l89./8в5
.в+
%К"
inputs         ▄@
p
к "*в'
 К
0         ▄@
Ъ л
8__inference_batch_normalization_1_layer_call_fn_13982027o9.8/@в=
6в3
-К*
inputs                  @
p 
к "%К"                  @л
8__inference_batch_normalization_1_layer_call_fn_13982040o89./@в=
6в3
-К*
inputs                  @
p
к "%К"                  @Ы
8__inference_batch_normalization_1_layer_call_fn_13982053_9.8/8в5
.в+
%К"
inputs         ▄@
p 
к "К         ▄@Ы
8__inference_batch_normalization_1_layer_call_fn_13982066_89./8в5
.в+
%К"
inputs         ▄@
p
к "К         ▄@╙
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13982246|;0:1@в=
6в3
-К*
inputs                  @
p 
к "2в/
(К%
0                  @
Ъ ╙
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13982280|:;01@в=
6в3
-К*
inputs                  @
p
к "2в/
(К%
0                  @
Ъ ├
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13982300l;0:18в5
.в+
%К"
inputs         ▄@
p 
к "*в'
 К
0         ▄@
Ъ ├
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13982334l:;018в5
.в+
%К"
inputs         ▄@
p
к "*в'
 К
0         ▄@
Ъ л
8__inference_batch_normalization_2_layer_call_fn_13982187o;0:1@в=
6в3
-К*
inputs                  @
p 
к "%К"                  @л
8__inference_batch_normalization_2_layer_call_fn_13982200o:;01@в=
6в3
-К*
inputs                  @
p
к "%К"                  @Ы
8__inference_batch_normalization_2_layer_call_fn_13982213_;0:18в5
.в+
%К"
inputs         ▄@
p 
к "К         ▄@Ы
8__inference_batch_normalization_2_layer_call_fn_13982226_:;018в5
.в+
%К"
inputs         ▄@
p
к "К         ▄@╣
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_13982622b=4<53в0
)в&
 К
inputs         T
p 
к "%в"
К
0         T
Ъ ╣
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_13982656b<=453в0
)в&
 К
inputs         T
p
к "%в"
К
0         T
Ъ С
8__inference_batch_normalization_3_layer_call_fn_13982589U=4<53в0
)в&
 К
inputs         T
p 
к "К         TС
8__inference_batch_normalization_3_layer_call_fn_13982602U<=453в0
)в&
 К
inputs         T
p
к "К         T╤
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_13981926|7,6-@в=
6в3
-К*
inputs                  @
p 
к "2в/
(К%
0                  @
Ъ ╤
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_13981960|67,-@в=
6в3
-К*
inputs                  @
p
к "2в/
(К%
0                  @
Ъ ┴
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_13981980l7,6-8в5
.в+
%К"
inputs         ▄@
p 
к "*в'
 К
0         ▄@
Ъ ┴
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_13982014l67,-8в5
.в+
%К"
inputs         ▄@
p
к "*в'
 К
0         ▄@
Ъ й
6__inference_batch_normalization_layer_call_fn_13981867o7,6-@в=
6в3
-К*
inputs                  @
p 
к "%К"                  @й
6__inference_batch_normalization_layer_call_fn_13981880o67,-@в=
6в3
-К*
inputs                  @
p
к "%К"                  @Щ
6__inference_batch_normalization_layer_call_fn_13981893_7,6-8в5
.в+
%К"
inputs         ▄@
p 
к "К         ▄@Щ
6__inference_batch_normalization_layer_call_fn_13981906_67,-8в5
.в+
%К"
inputs         ▄@
p
к "К         ▄@Ў
I__inference_concatenate_layer_call_and_return_conditional_losses_13982526и~в{
tвq
oЪl
"К
inputs/0         @
"К
inputs/1         @
"К
inputs/2         @
к "&в#
К
0         └
Ъ ╬
.__inference_concatenate_layer_call_fn_13982518Ы~в{
tвq
oЪl
"К
inputs/0         @
"К
inputs/1         @
"К
inputs/2         @
к "К         └п
M__inference_dense_1_dropout_layer_call_and_return_conditional_losses_13982541^4в1
*в'
!К
inputs         └
p 
к "&в#
К
0         └
Ъ п
M__inference_dense_1_dropout_layer_call_and_return_conditional_losses_13982545^4в1
*в'
!К
inputs         └
p
к "&в#
К
0         └
Ъ З
2__inference_dense_1_dropout_layer_call_fn_13982531Q4в1
*в'
!К
inputs         └
p 
к "К         └З
2__inference_dense_1_dropout_layer_call_fn_13982536Q4в1
*в'
!К
inputs         └
p
к "К         └ж
E__inference_dense_1_layer_call_and_return_conditional_losses_13982576]230в-
&в#
!К
inputs         └
к "%в"
К
0         T
Ъ ~
*__inference_dense_1_layer_call_fn_13982560P230в-
&в#
!К
inputs         └
к "К         Tм
P__inference_dense_activation_1_layer_call_and_return_conditional_losses_13982666X/в,
%в"
 К
inputs         T
к "%в"
К
0         T
Ъ Д
5__inference_dense_activation_1_layer_call_fn_13982661K/в,
%в"
 К
inputs         T
к "К         T╫
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_13982483{IвF
?в<
6К3
inputs'                           

 
к ".в+
$К!
0                  
Ъ ╜
X__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_13982489a8в5
.в+
%К"
inputs         ▄@

 
к "%в"
К
0         @
Ъ п
=__inference_global_average_pooling1d_1_layer_call_fn_13982472nIвF
?в<
6К3
inputs'                           

 
к "!К                  Х
=__inference_global_average_pooling1d_1_layer_call_fn_13982477T8в5
.в+
%К"
inputs         ▄@

 
к "К         @╫
X__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_13982505{IвF
?в<
6К3
inputs'                           

 
к ".в+
$К!
0                  
Ъ ╜
X__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_13982511a8в5
.в+
%К"
inputs         ▄@

 
к "%в"
К
0         @
Ъ п
=__inference_global_average_pooling1d_2_layer_call_fn_13982494nIвF
?в<
6К3
inputs'                           

 
к "!К                  Х
=__inference_global_average_pooling1d_2_layer_call_fn_13982499T8в5
.в+
%К"
inputs         ▄@

 
к "К         @╒
V__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_13982461{IвF
?в<
6К3
inputs'                           

 
к ".в+
$К!
0                  
Ъ ╗
V__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_13982467a8в5
.в+
%К"
inputs         ▄@

 
к "%в"
К
0         @
Ъ н
;__inference_global_average_pooling1d_layer_call_fn_13982450nIвF
?в<
6К3
inputs'                           

 
к "!К                  У
;__inference_global_average_pooling1d_layer_call_fn_13982455T8в5
.в+
%К"
inputs         ▄@

 
к "К         @=
__inference_loss_fn_0_13982677&в

в 
к "К =
__inference_loss_fn_1_13982688(в

в 
к "К =
__inference_loss_fn_2_13982699*в

в 
к "К =
__inference_loss_fn_3_139827102в

в 
к "К ╬
E__inference_model_1_layer_call_and_return_conditional_losses_13980471Д*+()&';0:19.8/7,6-23=4<5Aв>
7в4
*К'
left_inputs         ▄
p 

 
к "%в"
К
0         T
Ъ ╬
E__inference_model_1_layer_call_and_return_conditional_losses_13980548Д*+()&':;0189./67,-23<=45Aв>
7в4
*К'
left_inputs         ▄
p

 
к "%в"
К
0         T
Ъ ╚
E__inference_model_1_layer_call_and_return_conditional_losses_13980883*+()&';0:19.8/7,6-23=4<5<в9
2в/
%К"
inputs         ▄
p 

 
к "%в"
К
0         T
Ъ ╚
E__inference_model_1_layer_call_and_return_conditional_losses_13981130*+()&':;0189./67,-23<=45<в9
2в/
%К"
inputs         ▄
p

 
к "%в"
К
0         T
Ъ е
*__inference_model_1_layer_call_fn_13980158w*+()&';0:19.8/7,6-23=4<5Aв>
7в4
*К'
left_inputs         ▄
p 

 
к "К         Tе
*__inference_model_1_layer_call_fn_13980394w*+()&':;0189./67,-23<=45Aв>
7в4
*К'
left_inputs         ▄
p

 
к "К         Tа
*__inference_model_1_layer_call_fn_13980680r*+()&';0:19.8/7,6-23=4<5<в9
2в/
%К"
inputs         ▄
p 

 
к "К         Tа
*__inference_model_1_layer_call_fn_13980733r*+()&':;0189./67,-23<=45<в9
2в/
%К"
inputs         ▄
p

 
к "К         T╞
&__inference_signature_wrapper_13980627Ы*+()&';0:19.8/7,6-23=4<5HвE
в 
>к;
9
left_inputs*К'
left_inputs         ▄"5к2
0
	basemodel#К 
	basemodel         T╖
M__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_13981782f&'4в1
*в'
%К"
inputs         ▄
к "*в'
 К
0         ▄@
Ъ П
2__inference_stream_0_conv_1_layer_call_fn_13981761Y&'4в1
*в'
%К"
inputs         ▄
к "К         ▄@╖
M__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_13982379f8в5
.в+
%К"
inputs         ▄@
p 
к "*в'
 К
0         ▄@
Ъ ╖
M__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_13982391f8в5
.в+
%К"
inputs         ▄@
p
к "*в'
 К
0         ▄@
Ъ П
2__inference_stream_0_drop_1_layer_call_fn_13982369Y8в5
.в+
%К"
inputs         ▄@
p 
к "К         ▄@П
2__inference_stream_0_drop_1_layer_call_fn_13982374Y8в5
.в+
%К"
inputs         ▄@
p
к "К         ▄@╗
Q__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_13981680f8в5
.в+
%К"
inputs         ▄
p 
к "*в'
 К
0         ▄
Ъ ╗
Q__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_13981692f8в5
.в+
%К"
inputs         ▄
p
к "*в'
 К
0         ▄
Ъ У
6__inference_stream_0_input_drop_layer_call_fn_13981670Y8в5
.в+
%К"
inputs         ▄
p 
к "К         ▄У
6__inference_stream_0_input_drop_layer_call_fn_13981675Y8в5
.в+
%К"
inputs         ▄
p
к "К         ▄╖
M__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_13981818f()4в1
*в'
%К"
inputs         ▄
к "*в'
 К
0         ▄@
Ъ П
2__inference_stream_1_conv_1_layer_call_fn_13981797Y()4в1
*в'
%К"
inputs         ▄
к "К         ▄@╖
M__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_13982406f8в5
.в+
%К"
inputs         ▄@
p 
к "*в'
 К
0         ▄@
Ъ ╖
M__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_13982418f8в5
.в+
%К"
inputs         ▄@
p
к "*в'
 К
0         ▄@
Ъ П
2__inference_stream_1_drop_1_layer_call_fn_13982396Y8в5
.в+
%К"
inputs         ▄@
p 
к "К         ▄@П
2__inference_stream_1_drop_1_layer_call_fn_13982401Y8в5
.в+
%К"
inputs         ▄@
p
к "К         ▄@╗
Q__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_13981707f8в5
.в+
%К"
inputs         ▄
p 
к "*в'
 К
0         ▄
Ъ ╗
Q__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_13981719f8в5
.в+
%К"
inputs         ▄
p
к "*в'
 К
0         ▄
Ъ У
6__inference_stream_1_input_drop_layer_call_fn_13981697Y8в5
.в+
%К"
inputs         ▄
p 
к "К         ▄У
6__inference_stream_1_input_drop_layer_call_fn_13981702Y8в5
.в+
%К"
inputs         ▄
p
к "К         ▄╖
M__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_13981854f*+4в1
*в'
%К"
inputs         ▄
к "*в'
 К
0         ▄@
Ъ П
2__inference_stream_2_conv_1_layer_call_fn_13981833Y*+4в1
*в'
%К"
inputs         ▄
к "К         ▄@╖
M__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_13982433f8в5
.в+
%К"
inputs         ▄@
p 
к "*в'
 К
0         ▄@
Ъ ╖
M__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_13982445f8в5
.в+
%К"
inputs         ▄@
p
к "*в'
 К
0         ▄@
Ъ П
2__inference_stream_2_drop_1_layer_call_fn_13982423Y8в5
.в+
%К"
inputs         ▄@
p 
к "К         ▄@П
2__inference_stream_2_drop_1_layer_call_fn_13982428Y8в5
.в+
%К"
inputs         ▄@
p
к "К         ▄@╗
Q__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_13981734f8в5
.в+
%К"
inputs         ▄
p 
к "*в'
 К
0         ▄
Ъ ╗
Q__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_13981746f8в5
.в+
%К"
inputs         ▄
p
к "*в'
 К
0         ▄
Ъ У
6__inference_stream_2_input_drop_layer_call_fn_13981724Y8в5
.в+
%К"
inputs         ▄
p 
к "К         ▄У
6__inference_stream_2_input_drop_layer_call_fn_13981729Y8в5
.в+
%К"
inputs         ▄
p
к "К         ▄