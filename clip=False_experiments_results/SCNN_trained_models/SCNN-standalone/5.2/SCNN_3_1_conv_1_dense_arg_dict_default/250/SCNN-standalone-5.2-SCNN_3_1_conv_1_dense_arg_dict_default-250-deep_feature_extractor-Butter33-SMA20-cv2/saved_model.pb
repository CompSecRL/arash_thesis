й4
аІ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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

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

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

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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
list(type)(0
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
О
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.22v2.6.1-9-gc2363d6d0258ль/

stream_0_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_0_conv_1/kernel

*stream_0_conv_1/kernel/Read/ReadVariableOpReadVariableOpstream_0_conv_1/kernel*"
_output_shapes
:@*
dtype0

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

stream_1_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_1_conv_1/kernel

*stream_1_conv_1/kernel/Read/ReadVariableOpReadVariableOpstream_1_conv_1/kernel*"
_output_shapes
:@*
dtype0

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

stream_2_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_2_conv_1/kernel

*stream_2_conv_1/kernel/Read/ReadVariableOpReadVariableOpstream_2_conv_1/kernel*"
_output_shapes
:@*
dtype0

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

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:@*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:@*
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0

batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р@*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	Р@*
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

batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_3/gamma

/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:@*
dtype0

batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_3/beta

.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:@*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:@*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:@*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
Ђ
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
Ђ
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0

!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_3/moving_mean

5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
Ђ
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_3/moving_variance

9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0

NoOpNoOp
c
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Фb
valueКbBЗb BАb

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
Ў
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
layer-23
 layer-24
!layer-25
"layer_with_weights-6
"layer-26
#layer_with_weights-7
#layer-27
$layer-28
%regularization_losses
&trainable_variables
'	variables
(	keras_api
 
v
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
Ж
)0
*1
+2
,3
-4
.5
/6
07
98
:9
110
211
;12
<13
314
415
=16
>17
518
619
720
821
?22
@23
­
Anon_trainable_variables
Bmetrics
Clayer_regularization_losses
regularization_losses
Dlayer_metrics

Elayers
trainable_variables
	variables
 
 
 
 
R
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
R
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
R
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
h

)kernel
*bias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
h

+kernel
,bias
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
h

-kernel
.bias
Zregularization_losses
[trainable_variables
\	variables
]	keras_api

^axis
	/gamma
0beta
9moving_mean
:moving_variance
_regularization_losses
`trainable_variables
a	variables
b	keras_api

caxis
	1gamma
2beta
;moving_mean
<moving_variance
dregularization_losses
etrainable_variables
f	variables
g	keras_api

haxis
	3gamma
4beta
=moving_mean
>moving_variance
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
R
mregularization_losses
ntrainable_variables
o	variables
p	keras_api
R
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
R
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
R
yregularization_losses
ztrainable_variables
{	variables
|	keras_api
S
}regularization_losses
~trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
	keras_api
V
regularization_losses
trainable_variables
	variables
 	keras_api
V
Ёregularization_losses
Ђtrainable_variables
Ѓ	variables
Є	keras_api
l

5kernel
6bias
Ѕregularization_losses
Іtrainable_variables
Ї	variables
Ј	keras_api

	Љaxis
	7gamma
8beta
?moving_mean
@moving_variance
Њregularization_losses
Ћtrainable_variables
Ќ	variables
­	keras_api
V
Ўregularization_losses
Џtrainable_variables
А	variables
Б	keras_api
 
v
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
Ж
)0
*1
+2
,3
-4
.5
/6
07
98
:9
110
211
;12
<13
314
415
=16
>17
518
619
720
821
?22
@23
В
Вnon_trainable_variables
Гmetrics
 Дlayer_regularization_losses
%regularization_losses
Еlayer_metrics
Жlayers
&trainable_variables
'	variables
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
8
90
:1
;2
<3
=4
>5
?6
@7
 
 
 

0
1
 
 
 
В
Зmetrics
Иnon_trainable_variables
 Йlayer_regularization_losses
Fregularization_losses
Кlayer_metrics
Лlayers
Gtrainable_variables
H	variables
 
 
 
В
Мmetrics
Нnon_trainable_variables
 Оlayer_regularization_losses
Jregularization_losses
Пlayer_metrics
Рlayers
Ktrainable_variables
L	variables
 
 
 
В
Сmetrics
Тnon_trainable_variables
 Уlayer_regularization_losses
Nregularization_losses
Фlayer_metrics
Хlayers
Otrainable_variables
P	variables
 

)0
*1

)0
*1
В
Цmetrics
Чnon_trainable_variables
 Шlayer_regularization_losses
Rregularization_losses
Щlayer_metrics
Ъlayers
Strainable_variables
T	variables
 

+0
,1

+0
,1
В
Ыmetrics
Ьnon_trainable_variables
 Эlayer_regularization_losses
Vregularization_losses
Юlayer_metrics
Яlayers
Wtrainable_variables
X	variables
 

-0
.1

-0
.1
В
аmetrics
бnon_trainable_variables
 вlayer_regularization_losses
Zregularization_losses
гlayer_metrics
дlayers
[trainable_variables
\	variables
 
 

/0
01

/0
01
92
:3
В
еmetrics
жnon_trainable_variables
 зlayer_regularization_losses
_regularization_losses
иlayer_metrics
йlayers
`trainable_variables
a	variables
 
 

10
21

10
21
;2
<3
В
кmetrics
лnon_trainable_variables
 мlayer_regularization_losses
dregularization_losses
нlayer_metrics
оlayers
etrainable_variables
f	variables
 
 

30
41

30
41
=2
>3
В
пmetrics
рnon_trainable_variables
 сlayer_regularization_losses
iregularization_losses
тlayer_metrics
уlayers
jtrainable_variables
k	variables
 
 
 
В
фmetrics
хnon_trainable_variables
 цlayer_regularization_losses
mregularization_losses
чlayer_metrics
шlayers
ntrainable_variables
o	variables
 
 
 
В
щmetrics
ъnon_trainable_variables
 ыlayer_regularization_losses
qregularization_losses
ьlayer_metrics
эlayers
rtrainable_variables
s	variables
 
 
 
В
юmetrics
яnon_trainable_variables
 №layer_regularization_losses
uregularization_losses
ёlayer_metrics
ђlayers
vtrainable_variables
w	variables
 
 
 
В
ѓmetrics
єnon_trainable_variables
 ѕlayer_regularization_losses
yregularization_losses
іlayer_metrics
їlayers
ztrainable_variables
{	variables
 
 
 
В
јmetrics
љnon_trainable_variables
 њlayer_regularization_losses
}regularization_losses
ћlayer_metrics
ќlayers
~trainable_variables
	variables
 
 
 
Е
§metrics
ўnon_trainable_variables
 џlayer_regularization_losses
regularization_losses
layer_metrics
layers
trainable_variables
	variables
 
 
 
Е
metrics
non_trainable_variables
 layer_regularization_losses
regularization_losses
layer_metrics
layers
trainable_variables
	variables
 
 
 
Е
metrics
non_trainable_variables
 layer_regularization_losses
regularization_losses
layer_metrics
layers
trainable_variables
	variables
 
 
 
Е
metrics
non_trainable_variables
 layer_regularization_losses
regularization_losses
layer_metrics
layers
trainable_variables
	variables
 
 
 
Е
metrics
non_trainable_variables
 layer_regularization_losses
regularization_losses
layer_metrics
layers
trainable_variables
	variables
 
 
 
Е
metrics
non_trainable_variables
 layer_regularization_losses
regularization_losses
layer_metrics
layers
trainable_variables
	variables
 
 
 
Е
metrics
non_trainable_variables
 layer_regularization_losses
regularization_losses
layer_metrics
layers
trainable_variables
	variables
 
 
 
Е
 metrics
Ёnon_trainable_variables
 Ђlayer_regularization_losses
regularization_losses
Ѓlayer_metrics
Єlayers
trainable_variables
	variables
 
 
 
Е
Ѕmetrics
Іnon_trainable_variables
 Їlayer_regularization_losses
Ёregularization_losses
Јlayer_metrics
Љlayers
Ђtrainable_variables
Ѓ	variables
 

50
61

50
61
Е
Њmetrics
Ћnon_trainable_variables
 Ќlayer_regularization_losses
Ѕregularization_losses
­layer_metrics
Ўlayers
Іtrainable_variables
Ї	variables
 
 

70
81

70
81
?2
@3
Е
Џmetrics
Аnon_trainable_variables
 Бlayer_regularization_losses
Њregularization_losses
Вlayer_metrics
Гlayers
Ћtrainable_variables
Ќ	variables
 
 
 
Е
Дmetrics
Еnon_trainable_variables
 Жlayer_regularization_losses
Ўregularization_losses
Зlayer_metrics
Иlayers
Џtrainable_variables
А	variables
8
90
:1
;2
<3
=4
>5
?6
@7
 
 
 
о
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
"26
#27
$28
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
90
:1
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
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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

serving_default_left_inputsPlaceholder*,
_output_shapes
:џџџџџџџџџњ*
dtype0*!
shape:џџџџџџџџџњ

StatefulPartitionedCallStatefulPartitionedCallserving_default_left_inputsstream_2_conv_1/kernelstream_2_conv_1/biasstream_1_conv_1/kernelstream_1_conv_1/biasstream_0_conv_1/kernelstream_0_conv_1/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/beta%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/beta#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense_1/kerneldense_1/bias%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/beta*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_3597809
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
У
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
GPU2*0J 8 *)
f$R"
 __inference__traced_save_3600396
о
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
GPU2*0J 8 *,
f'R%
#__inference__traced_restore_3600478З.
Ќ
k
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_3599840

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@2

ExpandDims
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2	
MaxPool|
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ@:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
Ѓ
X
<__inference_global_average_pooling1d_2_layer_call_fn_3600044

inputs
identityс
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_35953402
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ы
а
5__inference_batch_normalization_layer_call_fn_3599474

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_35962132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџњ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
Ћн
Ј
F__inference_basemodel_layer_call_and_return_conditional_losses_3595957

inputs
inputs_1
inputs_2-
stream_2_conv_1_3595579:@%
stream_2_conv_1_3595581:@-
stream_1_conv_1_3595615:@%
stream_1_conv_1_3595617:@-
stream_0_conv_1_3595651:@%
stream_0_conv_1_3595653:@+
batch_normalization_2_3595676:@+
batch_normalization_2_3595678:@+
batch_normalization_2_3595680:@+
batch_normalization_2_3595682:@+
batch_normalization_1_3595705:@+
batch_normalization_1_3595707:@+
batch_normalization_1_3595709:@+
batch_normalization_1_3595711:@)
batch_normalization_3595734:@)
batch_normalization_3595736:@)
batch_normalization_3595738:@)
batch_normalization_3595740:@"
dense_1_3595876:	Р@
dense_1_3595878:@+
batch_normalization_3_3595881:@+
batch_normalization_3_3595883:@+
batch_normalization_3_3595885:@+
batch_normalization_3_3595887:@
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂ-dense_1/kernel/Regularizer/Abs/ReadVariableOpЂ0dense_1/kernel/Regularizer/Square/ReadVariableOpЂ'stream_0_conv_1/StatefulPartitionedCallЂ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ'stream_1_conv_1/StatefulPartitionedCallЂ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ'stream_2_conv_1/StatefulPartitionedCallЂ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp
#stream_2_input_drop/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_35955322%
#stream_2_input_drop/PartitionedCall
#stream_1_input_drop/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_35955392%
#stream_1_input_drop/PartitionedCallў
#stream_0_input_drop/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_35955462%
#stream_0_input_drop/PartitionedCallш
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_2_input_drop/PartitionedCall:output:0stream_2_conv_1_3595579stream_2_conv_1_3595581*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_35955782)
'stream_2_conv_1/StatefulPartitionedCallш
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_1_input_drop/PartitionedCall:output:0stream_1_conv_1_3595615stream_1_conv_1_3595617*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_35956142)
'stream_1_conv_1/StatefulPartitionedCallш
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_3595651stream_0_conv_1_3595653*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_35956502)
'stream_0_conv_1/StatefulPartitionedCallЬ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_3595676batch_normalization_2_3595678batch_normalization_2_3595680batch_normalization_2_3595682*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_35956752/
-batch_normalization_2/StatefulPartitionedCallЬ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_3595705batch_normalization_1_3595707batch_normalization_1_3595709batch_normalization_1_3595711*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35957042/
-batch_normalization_1/StatefulPartitionedCallО
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_3595734batch_normalization_3595736batch_normalization_3595738batch_normalization_3595740*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_35957332-
+batch_normalization/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_35957482
activation_2/PartitionedCall
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_35957552
activation_1/PartitionedCall
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_35957622
activation/PartitionedCall
"stream_2_maxpool_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_35957712$
"stream_2_maxpool_1/PartitionedCall
"stream_1_maxpool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_35957802$
"stream_1_maxpool_1/PartitionedCall
"stream_0_maxpool_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_35957892$
"stream_0_maxpool_1/PartitionedCall
stream_2_drop_1/PartitionedCallPartitionedCall+stream_2_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_35957962!
stream_2_drop_1/PartitionedCall
stream_1_drop_1/PartitionedCallPartitionedCall+stream_1_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_35958032!
stream_1_drop_1/PartitionedCall
stream_0_drop_1/PartitionedCallPartitionedCall+stream_0_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_35958102!
stream_0_drop_1/PartitionedCallЊ
(global_average_pooling1d/PartitionedCallPartitionedCall(stream_0_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_35958172*
(global_average_pooling1d/PartitionedCallА
*global_average_pooling1d_1/PartitionedCallPartitionedCall(stream_1_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_35958242,
*global_average_pooling1d_1/PartitionedCallА
*global_average_pooling1d_2/PartitionedCallPartitionedCall(stream_2_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_35958312,
*global_average_pooling1d_2/PartitionedCallљ
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:03global_average_pooling1d_1/PartitionedCall:output:03global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_35958412
concatenate/PartitionedCall
dense_1_dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_35958482!
dense_1_dropout/PartitionedCallЗ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_3595876dense_1_3595878*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_35958752!
dense_1/StatefulPartitionedCallП
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_3595881batch_normalization_3_3595883batch_normalization_3_3595885batch_normalization_3_3595887*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_35953782/
-batch_normalization_3/StatefulPartitionedCallІ
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_35958942$
"dense_activation_1/PartitionedCall
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/ConstЪ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_3595651*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs­
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1й
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_0_conv_1/kernel/Regularizer/mul/xм
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulй
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addа
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_1_3595651*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square­
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2р
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1и
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/ConstЪ
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_1_conv_1_3595615*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs­
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1й
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_1_conv_1/kernel/Regularizer/mul/xм
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulй
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addа
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_3595615*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square­
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2р
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1и
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/ConstЪ
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_3595579*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs­
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1й
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_2_conv_1/kernel/Regularizer/mul/xм
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulй
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addа
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_2_conv_1_3595579*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square­
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2р
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1и
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/ConstЏ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_3595876*
_output_shapes
:	Р@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpЈ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2 
dense_1/kernel/Regularizer/Abs
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1Й
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_1/kernel/Regularizer/mul/xМ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addЕ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_3595876*
_output_shapes
:	Р@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpД
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2#
!dense_1/kernel/Regularizer/Square
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2Р
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"dense_1/kernel/Regularizer/mul_1/xФ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1И
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityш
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_1_conv_1/StatefulPartitionedCall6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesz
x:џџџџџџџџџњ:џџџџџџџџџњ:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
'stream_1_conv_1/StatefulPartitionedCall'stream_1_conv_1/StatefulPartitionedCall2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2R
'stream_2_conv_1/StatefulPartitionedCall'stream_2_conv_1/StatefulPartitionedCall2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs:TP
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs:TP
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs

Б
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3595675

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

IdentityТ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџњ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
ы
e
I__inference_activation_2_layer_call_and_return_conditional_losses_3599819

inputs
identityS
TanhTanhinputs*
T0*,
_output_shapes
:џџџџџџџџџњ@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ@:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
џ

F__inference_basemodel_layer_call_and_return_conditional_losses_3598961
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
&dense_1_matmul_readvariableop_resource:	Р@5
'dense_1_biasadd_readvariableop_resource:@K
=batch_normalization_3_assignmovingavg_readvariableop_resource:@M
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_3_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_3_batchnorm_readvariableop_resource:@
identityЂ#batch_normalization/AssignMovingAvgЂ2batch_normalization/AssignMovingAvg/ReadVariableOpЂ%batch_normalization/AssignMovingAvg_1Ђ4batch_normalization/AssignMovingAvg_1/ReadVariableOpЂ,batch_normalization/batchnorm/ReadVariableOpЂ0batch_normalization/batchnorm/mul/ReadVariableOpЂ%batch_normalization_1/AssignMovingAvgЂ4batch_normalization_1/AssignMovingAvg/ReadVariableOpЂ'batch_normalization_1/AssignMovingAvg_1Ђ6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpЂ.batch_normalization_1/batchnorm/ReadVariableOpЂ2batch_normalization_1/batchnorm/mul/ReadVariableOpЂ%batch_normalization_2/AssignMovingAvgЂ4batch_normalization_2/AssignMovingAvg/ReadVariableOpЂ'batch_normalization_2/AssignMovingAvg_1Ђ6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpЂ.batch_normalization_2/batchnorm/ReadVariableOpЂ2batch_normalization_2/batchnorm/mul/ReadVariableOpЂ%batch_normalization_3/AssignMovingAvgЂ4batch_normalization_3/AssignMovingAvg/ReadVariableOpЂ'batch_normalization_3/AssignMovingAvg_1Ђ6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpЂ.batch_normalization_3/batchnorm/ReadVariableOpЂ2batch_normalization_3/batchnorm/mul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂ-dense_1/kernel/Regularizer/Abs/ReadVariableOpЂ0dense_1/kernel/Regularizer/Square/ReadVariableOpЂ&stream_0_conv_1/BiasAdd/ReadVariableOpЂ2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpЂ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ&stream_1_conv_1/BiasAdd/ReadVariableOpЂ2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpЂ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ&stream_2_conv_1/BiasAdd/ReadVariableOpЂ2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpЂ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp
!stream_2_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!stream_2_input_drop/dropout/ConstЖ
stream_2_input_drop/dropout/MulMulinputs_2*stream_2_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2!
stream_2_input_drop/dropout/Mul~
!stream_2_input_drop/dropout/ShapeShapeinputs_2*
T0*
_output_shapes
:2#
!stream_2_input_drop/dropout/Shape
8stream_2_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_2_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ*
dtype0*
seedЗ*
seed2Й2:
8stream_2_input_drop/dropout/random_uniform/RandomUniform
*stream_2_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2,
*stream_2_input_drop/dropout/GreaterEqual/y
(stream_2_input_drop/dropout/GreaterEqualGreaterEqualAstream_2_input_drop/dropout/random_uniform/RandomUniform:output:03stream_2_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2*
(stream_2_input_drop/dropout/GreaterEqualР
 stream_2_input_drop/dropout/CastCast,stream_2_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџњ2"
 stream_2_input_drop/dropout/CastЯ
!stream_2_input_drop/dropout/Mul_1Mul#stream_2_input_drop/dropout/Mul:z:0$stream_2_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџњ2#
!stream_2_input_drop/dropout/Mul_1
!stream_1_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!stream_1_input_drop/dropout/ConstЖ
stream_1_input_drop/dropout/MulMulinputs_1*stream_1_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2!
stream_1_input_drop/dropout/Mul~
!stream_1_input_drop/dropout/ShapeShapeinputs_1*
T0*
_output_shapes
:2#
!stream_1_input_drop/dropout/Shape
8stream_1_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_1_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ*
dtype0*
seedЗ*
seed2И2:
8stream_1_input_drop/dropout/random_uniform/RandomUniform
*stream_1_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2,
*stream_1_input_drop/dropout/GreaterEqual/y
(stream_1_input_drop/dropout/GreaterEqualGreaterEqualAstream_1_input_drop/dropout/random_uniform/RandomUniform:output:03stream_1_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2*
(stream_1_input_drop/dropout/GreaterEqualР
 stream_1_input_drop/dropout/CastCast,stream_1_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџњ2"
 stream_1_input_drop/dropout/CastЯ
!stream_1_input_drop/dropout/Mul_1Mul#stream_1_input_drop/dropout/Mul:z:0$stream_1_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџњ2#
!stream_1_input_drop/dropout/Mul_1
!stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!stream_0_input_drop/dropout/ConstЖ
stream_0_input_drop/dropout/MulMulinputs_0*stream_0_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2!
stream_0_input_drop/dropout/Mul~
!stream_0_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/Shape
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ*
dtype0*
seedЗ*
seed2З2:
8stream_0_input_drop/dropout/random_uniform/RandomUniform
*stream_0_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2,
*stream_0_input_drop/dropout/GreaterEqual/y
(stream_0_input_drop/dropout/GreaterEqualGreaterEqualAstream_0_input_drop/dropout/random_uniform/RandomUniform:output:03stream_0_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2*
(stream_0_input_drop/dropout/GreaterEqualР
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџњ2"
 stream_0_input_drop/dropout/CastЯ
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџњ2#
!stream_0_input_drop/dropout/Mul_1
%stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2'
%stream_2_conv_1/conv1d/ExpandDims/dimц
!stream_2_conv_1/conv1d/ExpandDims
ExpandDims%stream_2_input_drop/dropout/Mul_1:z:0.stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ2#
!stream_2_conv_1/conv1d/ExpandDimsш
2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp
'stream_2_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_2_conv_1/conv1d/ExpandDims_1/dimї
#stream_2_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_2_conv_1/conv1d/ExpandDims_1ї
stream_2_conv_1/conv1dConv2D*stream_2_conv_1/conv1d/ExpandDims:output:0,stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2
stream_2_conv_1/conv1dУ
stream_2_conv_1/conv1d/SqueezeSqueezestream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ2 
stream_2_conv_1/conv1d/SqueezeМ
&stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_2_conv_1/BiasAdd/ReadVariableOpЭ
stream_2_conv_1/BiasAddBiasAdd'stream_2_conv_1/conv1d/Squeeze:output:0.stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
stream_2_conv_1/BiasAdd
%stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2'
%stream_1_conv_1/conv1d/ExpandDims/dimц
!stream_1_conv_1/conv1d/ExpandDims
ExpandDims%stream_1_input_drop/dropout/Mul_1:z:0.stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ2#
!stream_1_conv_1/conv1d/ExpandDimsш
2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp
'stream_1_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_1_conv_1/conv1d/ExpandDims_1/dimї
#stream_1_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_1_conv_1/conv1d/ExpandDims_1ї
stream_1_conv_1/conv1dConv2D*stream_1_conv_1/conv1d/ExpandDims:output:0,stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2
stream_1_conv_1/conv1dУ
stream_1_conv_1/conv1d/SqueezeSqueezestream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ2 
stream_1_conv_1/conv1d/SqueezeМ
&stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_1_conv_1/BiasAdd/ReadVariableOpЭ
stream_1_conv_1/BiasAddBiasAdd'stream_1_conv_1/conv1d/Squeeze:output:0.stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
stream_1_conv_1/BiasAdd
%stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2'
%stream_0_conv_1/conv1d/ExpandDims/dimц
!stream_0_conv_1/conv1d/ExpandDims
ExpandDims%stream_0_input_drop/dropout/Mul_1:z:0.stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ2#
!stream_0_conv_1/conv1d/ExpandDimsш
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp
'stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_1/conv1d/ExpandDims_1/dimї
#stream_0_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1ї
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2
stream_0_conv_1/conv1dУ
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ2 
stream_0_conv_1/conv1d/SqueezeМ
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOpЭ
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
stream_0_conv_1/BiasAddН
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
"batch_normalization_2/moments/meanТ
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*"
_output_shapes
:@2,
*batch_normalization_2/moments/StopGradient
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference stream_2_conv_1/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@21
/batch_normalization_2/moments/SquaredDifferenceХ
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_2/moments/variance/reduction_indices
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2(
&batch_normalization_2/moments/varianceУ
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_2/moments/SqueezeЫ
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2-
+batch_normalization_2/AssignMovingAvg/decayц
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp№
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/subч
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/mul­
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_2/AssignMovingAvgЃ
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2/
-batch_normalization_2/AssignMovingAvg_1/decayь
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpј
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/subя
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/mulЗ
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_2/AssignMovingAvg_1
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_2/batchnorm/add/yк
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/addЅ
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpн
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/mulз
%batch_normalization_2/batchnorm/mul_1Mul stream_2_conv_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2'
%batch_normalization_2/batchnorm/mul_1г
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/mul_2д
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOpй
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/subт
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2'
%batch_normalization_2/batchnorm/add_1Н
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
"batch_normalization_1/moments/meanТ
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:@2,
*batch_normalization_1/moments/StopGradient
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_1_conv_1/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@21
/batch_normalization_1/moments/SquaredDifferenceХ
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_1/moments/variance/reduction_indices
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2(
&batch_normalization_1/moments/varianceУ
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_1/moments/SqueezeЫ
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2-
+batch_normalization_1/AssignMovingAvg/decayц
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp№
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_1/AssignMovingAvg/subч
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_1/AssignMovingAvg/mul­
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_1/AssignMovingAvgЃ
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2/
-batch_normalization_1/AssignMovingAvg_1/decayь
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpј
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_1/AssignMovingAvg_1/subя
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_1/AssignMovingAvg_1/mulЗ
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_1/AssignMovingAvg_1
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_1/batchnorm/add/yк
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/addЅ
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_1/batchnorm/Rsqrtр
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOpн
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/mulз
%batch_normalization_1/batchnorm/mul_1Mul stream_1_conv_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2'
%batch_normalization_1/batchnorm/mul_1г
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_1/batchnorm/mul_2д
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOpй
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/subт
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2'
%batch_normalization_1/batchnorm/add_1Й
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
 batch_normalization/moments/meanМ
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:@2*
(batch_normalization/moments/StopGradientџ
-batch_normalization/moments/SquaredDifferenceSquaredDifference stream_0_conv_1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2/
-batch_normalization/moments/SquaredDifferenceС
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       28
6batch_normalization/moments/variance/reduction_indices
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2&
$batch_normalization/moments/varianceН
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2%
#batch_normalization/moments/SqueezeХ
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2+
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
'batch_normalization/AssignMovingAvg/subп
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2)
'batch_normalization/AssignMovingAvg/mulЃ
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02%
#batch_normalization/AssignMovingAvg
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2-
+batch_normalization/AssignMovingAvg_1/decayц
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp№
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2+
)batch_normalization/AssignMovingAvg_1/subч
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2+
)batch_normalization/AssignMovingAvg_1/mul­
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization/AssignMovingAvg_1
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2%
#batch_normalization/batchnorm/add/yв
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/add
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/Rsqrtк
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpе
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/mulб
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2%
#batch_normalization/batchnorm/mul_1Ы
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/mul_2Ю
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpб
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/subк
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2%
#batch_normalization/batchnorm/add_1
activation_2/TanhTanh)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
activation_2/Tanh
activation_1/TanhTanh)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
activation_1/Tanh
activation/TanhTanh'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
activation/Tanh
!stream_2_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_2_maxpool_1/ExpandDims/dimЪ
stream_2_maxpool_1/ExpandDims
ExpandDimsactivation_2/Tanh:y:0*stream_2_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@2
stream_2_maxpool_1/ExpandDimsи
stream_2_maxpool_1/MaxPoolMaxPool&stream_2_maxpool_1/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2
stream_2_maxpool_1/MaxPoolЕ
stream_2_maxpool_1/SqueezeSqueeze#stream_2_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2
stream_2_maxpool_1/Squeeze
!stream_1_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_1_maxpool_1/ExpandDims/dimЪ
stream_1_maxpool_1/ExpandDims
ExpandDimsactivation_1/Tanh:y:0*stream_1_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@2
stream_1_maxpool_1/ExpandDimsи
stream_1_maxpool_1/MaxPoolMaxPool&stream_1_maxpool_1/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2
stream_1_maxpool_1/MaxPoolЕ
stream_1_maxpool_1/SqueezeSqueeze#stream_1_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2
stream_1_maxpool_1/Squeeze
!stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_1/ExpandDims/dimШ
stream_0_maxpool_1/ExpandDims
ExpandDimsactivation/Tanh:y:0*stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@2
stream_0_maxpool_1/ExpandDimsи
stream_0_maxpool_1/MaxPoolMaxPool&stream_0_maxpool_1/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_1/MaxPoolЕ
stream_0_maxpool_1/SqueezeSqueeze#stream_0_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2
stream_0_maxpool_1/Squeeze
stream_2_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2
stream_2_drop_1/dropout/ConstФ
stream_2_drop_1/dropout/MulMul#stream_2_maxpool_1/Squeeze:output:0&stream_2_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
stream_2_drop_1/dropout/Mul
stream_2_drop_1/dropout/ShapeShape#stream_2_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2
stream_2_drop_1/dropout/Shape
4stream_2_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_2_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
dtype0*
seedЗ*
seed2К26
4stream_2_drop_1/dropout/random_uniform/RandomUniform
&stream_2_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2(
&stream_2_drop_1/dropout/GreaterEqual/y
$stream_2_drop_1/dropout/GreaterEqualGreaterEqual=stream_2_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_2_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2&
$stream_2_drop_1/dropout/GreaterEqualГ
stream_2_drop_1/dropout/CastCast(stream_2_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ}@2
stream_2_drop_1/dropout/CastО
stream_2_drop_1/dropout/Mul_1Mulstream_2_drop_1/dropout/Mul:z:0 stream_2_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
stream_2_drop_1/dropout/Mul_1
stream_1_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2
stream_1_drop_1/dropout/ConstФ
stream_1_drop_1/dropout/MulMul#stream_1_maxpool_1/Squeeze:output:0&stream_1_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
stream_1_drop_1/dropout/Mul
stream_1_drop_1/dropout/ShapeShape#stream_1_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2
stream_1_drop_1/dropout/Shape
4stream_1_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_1_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
dtype0*
seedЗ*
seed2Й26
4stream_1_drop_1/dropout/random_uniform/RandomUniform
&stream_1_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2(
&stream_1_drop_1/dropout/GreaterEqual/y
$stream_1_drop_1/dropout/GreaterEqualGreaterEqual=stream_1_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_1_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2&
$stream_1_drop_1/dropout/GreaterEqualГ
stream_1_drop_1/dropout/CastCast(stream_1_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ}@2
stream_1_drop_1/dropout/CastО
stream_1_drop_1/dropout/Mul_1Mulstream_1_drop_1/dropout/Mul:z:0 stream_1_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
stream_1_drop_1/dropout/Mul_1
stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2
stream_0_drop_1/dropout/ConstФ
stream_0_drop_1/dropout/MulMul#stream_0_maxpool_1/Squeeze:output:0&stream_0_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
stream_0_drop_1/dropout/Mul
stream_0_drop_1/dropout/ShapeShape#stream_0_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/Shape
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
dtype0*
seedЗ*
seed2И26
4stream_0_drop_1/dropout/random_uniform/RandomUniform
&stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2(
&stream_0_drop_1/dropout/GreaterEqual/y
$stream_0_drop_1/dropout/GreaterEqualGreaterEqual=stream_0_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2&
$stream_0_drop_1/dropout/GreaterEqualГ
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ}@2
stream_0_drop_1/dropout/CastО
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
stream_0_drop_1/dropout/Mul_1Є
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indicesе
global_average_pooling1d/MeanMean!stream_0_drop_1/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
global_average_pooling1d/MeanЈ
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_1/Mean/reduction_indicesл
global_average_pooling1d_1/MeanMean!stream_1_drop_1/dropout/Mul_1:z:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2!
global_average_pooling1d_1/MeanЈ
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indicesл
global_average_pooling1d_2/MeanMean!stream_2_drop_1/dropout/Mul_1:z:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2!
global_average_pooling1d_2/Meant
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis
concatenate/concatConcatV2&global_average_pooling1d/Mean:output:0(global_average_pooling1d_1/Mean:output:0(global_average_pooling1d_2/Mean:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџР2
concatenate/concatІ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02
dense_1/MatMul/ReadVariableOp 
dense_1/MatMulMatMulconcatenate/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_1/MatMulЄ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOpЁ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_1/BiasAddЖ
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indicesу
"batch_normalization_3/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2$
"batch_normalization_3/moments/meanО
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:@2,
*batch_normalization_3/moments/StopGradientј
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@21
/batch_normalization_3/moments/SquaredDifferenceО
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_3/moments/variance/reduction_indices
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2(
&batch_normalization_3/moments/varianceТ
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_3/moments/SqueezeЪ
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1
+batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2-
+batch_normalization_3/AssignMovingAvg/decayц
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp№
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_3/AssignMovingAvg/subч
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_3/AssignMovingAvg/mul­
%batch_normalization_3/AssignMovingAvgAssignSubVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_3/AssignMovingAvgЃ
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2/
-batch_normalization_3/AssignMovingAvg_1/decayь
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpј
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_3/AssignMovingAvg_1/subя
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_3/AssignMovingAvg_1/mulЗ
'batch_normalization_3/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_3/AssignMovingAvg_1
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_3/batchnorm/add/yк
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/addЅ
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/Rsqrtр
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpн
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/mulЪ
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2'
%batch_normalization_3/batchnorm/mul_1г
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/mul_2д
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpй
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/subн
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2'
%batch_normalization_3/batchnorm/add_1
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constю
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs­
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1й
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_0_conv_1/kernel/Regularizer/mul/xм
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulй
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addє
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square­
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2р
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1и
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Constю
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs­
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1й
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_1_conv_1/kernel/Regularizer/mul/xм
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulй
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addє
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square­
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2р
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1и
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Constю
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs­
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1й
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_2_conv_1/kernel/Regularizer/mul/xм
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulй
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addє
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square­
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2р
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1и
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/ConstЦ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpЈ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2 
dense_1/kernel/Regularizer/Abs
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1Й
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_1/kernel/Regularizer/mul/xМ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addЬ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpД
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2#
!dense_1/kernel/Regularizer/Square
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2Р
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"dense_1/kernel/Regularizer/mul_1/xФ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1И
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1
IdentityIdentity)batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityљ
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_1_conv_1/BiasAdd/ReadVariableOp3^stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_2_conv_1/BiasAdd/ReadVariableOp3^stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesz
x:џџџџџџџџџњ:џџџџџџџџџњ:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 2J
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
&stream_1_conv_1/BiasAdd/ReadVariableOp&stream_1_conv_1/BiasAdd/ReadVariableOp2h
2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2P
&stream_2_conv_1/BiasAdd/ReadVariableOp&stream_2_conv_1/BiasAdd/ReadVariableOp2h
2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:V R
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs/2
ы
e
I__inference_activation_2_layer_call_and_return_conditional_losses_3595748

inputs
identityS
TanhTanhinputs*
T0*,
_output_shapes
:џџџџџџџџџњ@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ@:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs

V
:__inference_global_average_pooling1d_layer_call_fn_3600000

inputs
identityп
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_35952922
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
С
j
1__inference_stream_0_drop_1_layer_call_fn_3599929

inputs
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_35960822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
к
в
7__inference_batch_normalization_3_layer_call_fn_3600212

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_35954382
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ы
e
I__inference_activation_1_layer_call_and_return_conditional_losses_3599809

inputs
identityS
TanhTanhinputs*
T0*,
_output_shapes
:џџџџџџџџџњ@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ@:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs

т
)__inference_model_1_layer_call_fn_3598406

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

unknown_17:	Р@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityЂStatefulPartitionedCall
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
:џџџџџџџџџ@*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_35973642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
ю
k
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_3599946

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeг
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
dtype0*
seedЗ*
seed2Й2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yТ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ}@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs

Ђ
1__inference_stream_0_conv_1_layer_call_fn_3599206

inputs
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_35956502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџњ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
О

+__inference_basemodel_layer_call_fn_3599016
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

unknown_17:	Р@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityЂStatefulPartitionedCallМ
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
:џџџџџџџџџ@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_35959572
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesz
x:џџџџџџџџџњ:џџџџџџџџџњ:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs/2
е
P
4__inference_dense_activation_1_layer_call_fn_3600221

inputs
identityа
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_35958942
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Л
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3595292

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
:џџџџџџџџџџџџџџџџџџ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
И
Б
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3595060

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2

IdentityТ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
С
j
1__inference_stream_1_drop_1_layer_call_fn_3599956

inputs
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_35961052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
ыш
В
F__inference_basemodel_layer_call_and_return_conditional_losses_3597028
inputs_0
inputs_1
inputs_2-
stream_2_conv_1_3596896:@%
stream_2_conv_1_3596898:@-
stream_1_conv_1_3596901:@%
stream_1_conv_1_3596903:@-
stream_0_conv_1_3596906:@%
stream_0_conv_1_3596908:@+
batch_normalization_2_3596911:@+
batch_normalization_2_3596913:@+
batch_normalization_2_3596915:@+
batch_normalization_2_3596917:@+
batch_normalization_1_3596920:@+
batch_normalization_1_3596922:@+
batch_normalization_1_3596924:@+
batch_normalization_1_3596926:@)
batch_normalization_3596929:@)
batch_normalization_3596931:@)
batch_normalization_3596933:@)
batch_normalization_3596935:@"
dense_1_3596952:	Р@
dense_1_3596954:@+
batch_normalization_3_3596957:@+
batch_normalization_3_3596959:@+
batch_normalization_3_3596961:@+
batch_normalization_3_3596963:@
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂ-dense_1/kernel/Regularizer/Abs/ReadVariableOpЂ0dense_1/kernel/Regularizer/Square/ReadVariableOpЂ'stream_0_conv_1/StatefulPartitionedCallЂ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ'stream_0_drop_1/StatefulPartitionedCallЂ+stream_0_input_drop/StatefulPartitionedCallЂ'stream_1_conv_1/StatefulPartitionedCallЂ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ'stream_1_drop_1/StatefulPartitionedCallЂ+stream_1_input_drop/StatefulPartitionedCallЂ'stream_2_conv_1/StatefulPartitionedCallЂ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ'stream_2_drop_1/StatefulPartitionedCallЂ+stream_2_input_drop/StatefulPartitionedCall
+stream_2_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_35964402-
+stream_2_input_drop/StatefulPartitionedCallЦ
+stream_1_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_1,^stream_2_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_35964172-
+stream_1_input_drop/StatefulPartitionedCallЦ
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_0,^stream_1_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_35963942-
+stream_0_input_drop/StatefulPartitionedCall№
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_2_input_drop/StatefulPartitionedCall:output:0stream_2_conv_1_3596896stream_2_conv_1_3596898*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_35955782)
'stream_2_conv_1/StatefulPartitionedCall№
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_1_input_drop/StatefulPartitionedCall:output:0stream_1_conv_1_3596901stream_1_conv_1_3596903*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_35956142)
'stream_1_conv_1/StatefulPartitionedCall№
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_3596906stream_0_conv_1_3596908*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_35956502)
'stream_0_conv_1/StatefulPartitionedCallЪ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_3596911batch_normalization_2_3596913batch_normalization_2_3596915batch_normalization_2_3596917*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_35963332/
-batch_normalization_2/StatefulPartitionedCallЪ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_3596920batch_normalization_1_3596922batch_normalization_1_3596924batch_normalization_1_3596926*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35962732/
-batch_normalization_1/StatefulPartitionedCallМ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_3596929batch_normalization_3596931batch_normalization_3596933batch_normalization_3596935*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_35962132-
+batch_normalization/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_35957482
activation_2/PartitionedCall
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_35957552
activation_1/PartitionedCall
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_35957622
activation/PartitionedCall
"stream_2_maxpool_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_35957712$
"stream_2_maxpool_1/PartitionedCall
"stream_1_maxpool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_35957802$
"stream_1_maxpool_1/PartitionedCall
"stream_0_maxpool_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_35957892$
"stream_0_maxpool_1/PartitionedCallм
'stream_2_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_2_maxpool_1/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_35961282)
'stream_2_drop_1/StatefulPartitionedCallи
'stream_1_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_1_maxpool_1/PartitionedCall:output:0(^stream_2_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_35961052)
'stream_1_drop_1/StatefulPartitionedCallи
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_0_maxpool_1/PartitionedCall:output:0(^stream_1_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_35960822)
'stream_0_drop_1/StatefulPartitionedCallВ
(global_average_pooling1d/PartitionedCallPartitionedCall0stream_0_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_35958172*
(global_average_pooling1d/PartitionedCallИ
*global_average_pooling1d_1/PartitionedCallPartitionedCall0stream_1_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_35958242,
*global_average_pooling1d_1/PartitionedCallИ
*global_average_pooling1d_2/PartitionedCallPartitionedCall0stream_2_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_35958312,
*global_average_pooling1d_2/PartitionedCallљ
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:03global_average_pooling1d_1/PartitionedCall:output:03global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_35958412
concatenate/PartitionedCall
dense_1_dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_35960362!
dense_1_dropout/PartitionedCallЗ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_3596952dense_1_3596954*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_35958752!
dense_1/StatefulPartitionedCallН
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_3596957batch_normalization_3_3596959batch_normalization_3_3596961batch_normalization_3_3596963*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_35954382/
-batch_normalization_3/StatefulPartitionedCallІ
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_35958942$
"dense_activation_1/PartitionedCall
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/ConstЪ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_3596906*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs­
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1й
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_0_conv_1/kernel/Regularizer/mul/xм
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulй
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addа
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_1_3596906*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square­
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2р
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1и
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/ConstЪ
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_1_conv_1_3596901*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs­
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1й
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_1_conv_1/kernel/Regularizer/mul/xм
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulй
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addа
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_3596901*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square­
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2р
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1и
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/ConstЪ
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_3596896*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs­
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1й
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_2_conv_1/kernel/Regularizer/mul/xм
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulй
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addа
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_2_conv_1_3596896*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square­
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2р
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1и
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/ConstЏ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_3596952*
_output_shapes
:	Р@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpЈ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2 
dense_1/kernel/Regularizer/Abs
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1Й
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_1/kernel/Regularizer/mul/xМ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addЕ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_3596952*
_output_shapes
:	Р@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpД
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2#
!dense_1/kernel/Regularizer/Square
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2Р
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"dense_1/kernel/Regularizer/mul_1/xФ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1И
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity№
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall(^stream_1_conv_1/StatefulPartitionedCall6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_1_drop_1/StatefulPartitionedCall,^stream_1_input_drop/StatefulPartitionedCall(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_2_drop_1/StatefulPartitionedCall,^stream_2_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesz
x:џџџџџџџџџњ:џџџџџџџџџњ:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
'stream_0_drop_1/StatefulPartitionedCall'stream_0_drop_1/StatefulPartitionedCall2Z
+stream_0_input_drop/StatefulPartitionedCall+stream_0_input_drop/StatefulPartitionedCall2R
'stream_1_conv_1/StatefulPartitionedCall'stream_1_conv_1/StatefulPartitionedCall2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2R
'stream_1_drop_1/StatefulPartitionedCall'stream_1_drop_1/StatefulPartitionedCall2Z
+stream_1_input_drop/StatefulPartitionedCall+stream_1_input_drop/StatefulPartitionedCall2R
'stream_2_conv_1/StatefulPartitionedCall'stream_2_conv_1/StatefulPartitionedCall2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp2R
'stream_2_drop_1/StatefulPartitionedCall'stream_2_drop_1/StatefulPartitionedCall2Z
+stream_2_input_drop/StatefulPartitionedCall+stream_2_input_drop/StatefulPartitionedCall:V R
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs_2
ё
в
7__inference_batch_normalization_1_layer_call_fn_3599621

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35957042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџњ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
њ
o
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_3599142

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeд
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ*
dtype0*
seedЗ*
seed2Й2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yУ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџњ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџњ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs

k
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_3595210

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

ExpandDimsБ
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ч
P
4__inference_stream_2_maxpool_1_layer_call_fn_3599902

inputs
identityд
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_35957712
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ@:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs

n
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_3595539

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
њ,

L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_3595650

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂ"conv1d/ExpandDims_1/ReadVariableOpЂ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ2
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
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2	
BiasAdd
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constо
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs­
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1й
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_0_conv_1/kernel/Regularizer/mul/xм
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulй
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addф
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square­
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2р
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1и
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1p
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityџ
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџњ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs

s
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_3595831

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
:џџџџџџџџџ@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
Ж
Џ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3594736

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2

IdentityТ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Э*
ы
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3595438

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
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
moments/StopGradientЄ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indicesВ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
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
з#<2
AssignMovingAvg/decayЄ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulП
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
з#<2
AssignMovingAvg_1/decayЊ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mulЩ
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityђ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ј
љ
__inference_loss_fn_0_3600241T
>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource:@
identityЂ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constё
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs­
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1й
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_0_conv_1/kernel/Regularizer/mul/xм
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulй
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addї
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square­
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2р
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1и
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1v
IdentityIdentity,stream_0_conv_1/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityС
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
ю
k
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_3596105

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeг
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
dtype0*
seedЗ*
seed2Й2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yТ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ}@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
Ё
т
)__inference_model_1_layer_call_fn_3598353

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

unknown_17:	Р@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityЂStatefulPartitionedCallЂ
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
:џџџџџџџџџ@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_35971452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
Ќ
k
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_3595789

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@2

ExpandDims
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2	
MaxPool|
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ@:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs

k
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_3595238

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

ExpandDimsБ
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
щ
V
:__inference_global_average_pooling1d_layer_call_fn_3600005

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_35958172
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
я
в
7__inference_batch_normalization_1_layer_call_fn_3599634

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35962732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџњ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
Ѓ
X
<__inference_global_average_pooling1d_1_layer_call_fn_3600022

inputs
identityс
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_35953162
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3595817

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
:џџџџџџџџџ@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
Ы
"
D__inference_model_1_layer_call_and_return_conditional_losses_3598300

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
0basemodel_dense_1_matmul_readvariableop_resource:	Р@?
1basemodel_dense_1_biasadd_readvariableop_resource:@U
Gbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource:@W
Ibasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource:@S
Ebasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:@O
Abasemodel_batch_normalization_3_batchnorm_readvariableop_resource:@
identityЂ-basemodel/batch_normalization/AssignMovingAvgЂ<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOpЂ/basemodel/batch_normalization/AssignMovingAvg_1Ђ>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOpЂ6basemodel/batch_normalization/batchnorm/ReadVariableOpЂ:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpЂ/basemodel/batch_normalization_1/AssignMovingAvgЂ>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOpЂ1basemodel/batch_normalization_1/AssignMovingAvg_1Ђ@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpЂ8basemodel/batch_normalization_1/batchnorm/ReadVariableOpЂ<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpЂ/basemodel/batch_normalization_2/AssignMovingAvgЂ>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOpЂ1basemodel/batch_normalization_2/AssignMovingAvg_1Ђ@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpЂ8basemodel/batch_normalization_2/batchnorm/ReadVariableOpЂ<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpЂ/basemodel/batch_normalization_3/AssignMovingAvgЂ>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOpЂ1basemodel/batch_normalization_3/AssignMovingAvg_1Ђ@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpЂ8basemodel/batch_normalization_3/batchnorm/ReadVariableOpЂ<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpЂ(basemodel/dense_1/BiasAdd/ReadVariableOpЂ'basemodel/dense_1/MatMul/ReadVariableOpЂ0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpЂ<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpЂ0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpЂ<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpЂ0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpЂ<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpЂ-dense_1/kernel/Regularizer/Abs/ReadVariableOpЂ0dense_1/kernel/Regularizer/Square/ReadVariableOpЂ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp
+basemodel/stream_2_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2-
+basemodel/stream_2_input_drop/dropout/Constв
)basemodel/stream_2_input_drop/dropout/MulMulinputs4basemodel/stream_2_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2+
)basemodel/stream_2_input_drop/dropout/Mul
+basemodel/stream_2_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2-
+basemodel/stream_2_input_drop/dropout/ShapeЎ
Bbasemodel/stream_2_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_2_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ*
dtype0*
seedЗ*
seed2Й2D
Bbasemodel/stream_2_input_drop/dropout/random_uniform/RandomUniformБ
4basemodel/stream_2_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>26
4basemodel/stream_2_input_drop/dropout/GreaterEqual/yЛ
2basemodel/stream_2_input_drop/dropout/GreaterEqualGreaterEqualKbasemodel/stream_2_input_drop/dropout/random_uniform/RandomUniform:output:0=basemodel/stream_2_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ24
2basemodel/stream_2_input_drop/dropout/GreaterEqualо
*basemodel/stream_2_input_drop/dropout/CastCast6basemodel/stream_2_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџњ2,
*basemodel/stream_2_input_drop/dropout/Castї
+basemodel/stream_2_input_drop/dropout/Mul_1Mul-basemodel/stream_2_input_drop/dropout/Mul:z:0.basemodel/stream_2_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџњ2-
+basemodel/stream_2_input_drop/dropout/Mul_1
+basemodel/stream_1_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2-
+basemodel/stream_1_input_drop/dropout/Constв
)basemodel/stream_1_input_drop/dropout/MulMulinputs4basemodel/stream_1_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2+
)basemodel/stream_1_input_drop/dropout/Mul
+basemodel/stream_1_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2-
+basemodel/stream_1_input_drop/dropout/ShapeЎ
Bbasemodel/stream_1_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_1_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ*
dtype0*
seedЗ*
seed2И2D
Bbasemodel/stream_1_input_drop/dropout/random_uniform/RandomUniformБ
4basemodel/stream_1_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>26
4basemodel/stream_1_input_drop/dropout/GreaterEqual/yЛ
2basemodel/stream_1_input_drop/dropout/GreaterEqualGreaterEqualKbasemodel/stream_1_input_drop/dropout/random_uniform/RandomUniform:output:0=basemodel/stream_1_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ24
2basemodel/stream_1_input_drop/dropout/GreaterEqualо
*basemodel/stream_1_input_drop/dropout/CastCast6basemodel/stream_1_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџњ2,
*basemodel/stream_1_input_drop/dropout/Castї
+basemodel/stream_1_input_drop/dropout/Mul_1Mul-basemodel/stream_1_input_drop/dropout/Mul:z:0.basemodel/stream_1_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџњ2-
+basemodel/stream_1_input_drop/dropout/Mul_1
+basemodel/stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2-
+basemodel/stream_0_input_drop/dropout/Constв
)basemodel/stream_0_input_drop/dropout/MulMulinputs4basemodel/stream_0_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2+
)basemodel/stream_0_input_drop/dropout/Mul
+basemodel/stream_0_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2-
+basemodel/stream_0_input_drop/dropout/ShapeЎ
Bbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ*
dtype0*
seedЗ*
seed2З2D
Bbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniformБ
4basemodel/stream_0_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>26
4basemodel/stream_0_input_drop/dropout/GreaterEqual/yЛ
2basemodel/stream_0_input_drop/dropout/GreaterEqualGreaterEqualKbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniform:output:0=basemodel/stream_0_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ24
2basemodel/stream_0_input_drop/dropout/GreaterEqualо
*basemodel/stream_0_input_drop/dropout/CastCast6basemodel/stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџњ2,
*basemodel/stream_0_input_drop/dropout/Castї
+basemodel/stream_0_input_drop/dropout/Mul_1Mul-basemodel/stream_0_input_drop/dropout/Mul:z:0.basemodel/stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџњ2-
+basemodel/stream_0_input_drop/dropout/Mul_1­
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ21
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dim
+basemodel/stream_2_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_2_input_drop/dropout/Mul_1:z:08basemodel/stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ2-
+basemodel/stream_2_conv_1/conv1d/ExpandDims
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpЈ
1basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dim
-basemodel/stream_2_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_2_conv_1/conv1d/ExpandDims_1
 basemodel/stream_2_conv_1/conv1dConv2D4basemodel/stream_2_conv_1/conv1d/ExpandDims:output:06basemodel/stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2"
 basemodel/stream_2_conv_1/conv1dс
(basemodel/stream_2_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ2*
(basemodel/stream_2_conv_1/conv1d/Squeezeк
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpѕ
!basemodel/stream_2_conv_1/BiasAddBiasAdd1basemodel/stream_2_conv_1/conv1d/Squeeze:output:08basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2#
!basemodel/stream_2_conv_1/BiasAdd­
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ21
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dim
+basemodel/stream_1_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_1_input_drop/dropout/Mul_1:z:08basemodel/stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ2-
+basemodel/stream_1_conv_1/conv1d/ExpandDims
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpЈ
1basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dim
-basemodel/stream_1_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_1_conv_1/conv1d/ExpandDims_1
 basemodel/stream_1_conv_1/conv1dConv2D4basemodel/stream_1_conv_1/conv1d/ExpandDims:output:06basemodel/stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2"
 basemodel/stream_1_conv_1/conv1dс
(basemodel/stream_1_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ2*
(basemodel/stream_1_conv_1/conv1d/Squeezeк
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpѕ
!basemodel/stream_1_conv_1/BiasAddBiasAdd1basemodel/stream_1_conv_1/conv1d/Squeeze:output:08basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2#
!basemodel/stream_1_conv_1/BiasAdd­
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ21
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dim
+basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_0_input_drop/dropout/Mul_1:z:08basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ2-
+basemodel/stream_0_conv_1/conv1d/ExpandDims
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpЈ
1basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1
 basemodel/stream_0_conv_1/conv1dConv2D4basemodel/stream_0_conv_1/conv1d/ExpandDims:output:06basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1dс
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ2*
(basemodel/stream_0_conv_1/conv1d/Squeezeк
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpѕ
!basemodel/stream_0_conv_1/BiasAddBiasAdd1basemodel/stream_0_conv_1/conv1d/Squeeze:output:08basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2#
!basemodel/stream_0_conv_1/BiasAddб
>basemodel/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>basemodel/batch_normalization_2/moments/mean/reduction_indices
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
4basemodel/batch_normalization_2/moments/StopGradient­
9basemodel/batch_normalization_2/moments/SquaredDifferenceSquaredDifference*basemodel/stream_2_conv_1/BiasAdd:output:0=basemodel/batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2;
9basemodel/batch_normalization_2/moments/SquaredDifferenceй
Bbasemodel/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bbasemodel/batch_normalization_2/moments/variance/reduction_indicesЖ
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
1basemodel/batch_normalization_2/moments/Squeeze_1Г
5basemodel/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<27
5basemodel/batch_normalization_2/AssignMovingAvg/decay
>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp
3basemodel/batch_normalization_2/AssignMovingAvg/subSubFbasemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:08basemodel/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization_2/AssignMovingAvg/sub
3basemodel/batch_normalization_2/AssignMovingAvg/mulMul7basemodel/batch_normalization_2/AssignMovingAvg/sub:z:0>basemodel/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization_2/AssignMovingAvg/mulп
/basemodel/batch_normalization_2/AssignMovingAvgAssignSubVariableOpGbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource7basemodel/batch_normalization_2/AssignMovingAvg/mul:z:0?^basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization_2/AssignMovingAvgЗ
7basemodel/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<29
7basemodel/batch_normalization_2/AssignMovingAvg_1/decay
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp 
5basemodel/batch_normalization_2/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_2/AssignMovingAvg_1/sub
5basemodel/batch_normalization_2/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_2/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_2/AssignMovingAvg_1/mulщ
1basemodel/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_2/AssignMovingAvg_1/mul:z:0A^basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_2/AssignMovingAvg_1Ї
/basemodel/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:21
/basemodel/batch_normalization_2/batchnorm/add/y
-basemodel/batch_normalization_2/batchnorm/addAddV2:basemodel/batch_normalization_2/moments/Squeeze_1:output:08basemodel/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/addУ
/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm/Rsqrtў
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp
-basemodel/batch_normalization_2/batchnorm/mulMul3basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/mulџ
/basemodel/batch_normalization_2/batchnorm/mul_1Mul*basemodel/stream_2_conv_1/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@21
/basemodel/batch_normalization_2/batchnorm/mul_1ћ
/basemodel/batch_normalization_2/batchnorm/mul_2Mul8basemodel/batch_normalization_2/moments/Squeeze:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm/mul_2ђ
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_2/batchnorm/ReadVariableOp
-basemodel/batch_normalization_2/batchnorm/subSub@basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/sub
/basemodel/batch_normalization_2/batchnorm/add_1AddV23basemodel/batch_normalization_2/batchnorm/mul_1:z:01basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@21
/basemodel/batch_normalization_2/batchnorm/add_1б
>basemodel/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>basemodel/batch_normalization_1/moments/mean/reduction_indices
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
4basemodel/batch_normalization_1/moments/StopGradient­
9basemodel/batch_normalization_1/moments/SquaredDifferenceSquaredDifference*basemodel/stream_1_conv_1/BiasAdd:output:0=basemodel/batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2;
9basemodel/batch_normalization_1/moments/SquaredDifferenceй
Bbasemodel/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bbasemodel/batch_normalization_1/moments/variance/reduction_indicesЖ
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
1basemodel/batch_normalization_1/moments/Squeeze_1Г
5basemodel/batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<27
5basemodel/batch_normalization_1/AssignMovingAvg/decay
>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp
3basemodel/batch_normalization_1/AssignMovingAvg/subSubFbasemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:08basemodel/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization_1/AssignMovingAvg/sub
3basemodel/batch_normalization_1/AssignMovingAvg/mulMul7basemodel/batch_normalization_1/AssignMovingAvg/sub:z:0>basemodel/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization_1/AssignMovingAvg/mulп
/basemodel/batch_normalization_1/AssignMovingAvgAssignSubVariableOpGbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource7basemodel/batch_normalization_1/AssignMovingAvg/mul:z:0?^basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization_1/AssignMovingAvgЗ
7basemodel/batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<29
7basemodel/batch_normalization_1/AssignMovingAvg_1/decay
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp 
5basemodel/batch_normalization_1/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_1/AssignMovingAvg_1/sub
5basemodel/batch_normalization_1/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_1/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_1/AssignMovingAvg_1/mulщ
1basemodel/batch_normalization_1/AssignMovingAvg_1AssignSubVariableOpIbasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_1/AssignMovingAvg_1/mul:z:0A^basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_1/AssignMovingAvg_1Ї
/basemodel/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:21
/basemodel/batch_normalization_1/batchnorm/add/y
-basemodel/batch_normalization_1/batchnorm/addAddV2:basemodel/batch_normalization_1/moments/Squeeze_1:output:08basemodel/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/addУ
/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm/Rsqrtў
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp
-basemodel/batch_normalization_1/batchnorm/mulMul3basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/mulџ
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_1_conv_1/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@21
/basemodel/batch_normalization_1/batchnorm/mul_1ћ
/basemodel/batch_normalization_1/batchnorm/mul_2Mul8basemodel/batch_normalization_1/moments/Squeeze:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm/mul_2ђ
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_1/batchnorm/ReadVariableOp
-basemodel/batch_normalization_1/batchnorm/subSub@basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/sub
/basemodel/batch_normalization_1/batchnorm/add_1AddV23basemodel/batch_normalization_1/batchnorm/mul_1:z:01basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@21
/basemodel/batch_normalization_1/batchnorm/add_1Э
<basemodel/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2>
<basemodel/batch_normalization/moments/mean/reduction_indices
*basemodel/batch_normalization/moments/meanMean*basemodel/stream_0_conv_1/BiasAdd:output:0Ebasemodel/batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2,
*basemodel/batch_normalization/moments/meanк
2basemodel/batch_normalization/moments/StopGradientStopGradient3basemodel/batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:@24
2basemodel/batch_normalization/moments/StopGradientЇ
7basemodel/batch_normalization/moments/SquaredDifferenceSquaredDifference*basemodel/stream_0_conv_1/BiasAdd:output:0;basemodel/batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@29
7basemodel/batch_normalization/moments/SquaredDifferenceе
@basemodel/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2B
@basemodel/batch_normalization/moments/variance/reduction_indicesЎ
.basemodel/batch_normalization/moments/varianceMean;basemodel/batch_normalization/moments/SquaredDifference:z:0Ibasemodel/batch_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(20
.basemodel/batch_normalization/moments/varianceл
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
/basemodel/batch_normalization/moments/Squeeze_1Џ
3basemodel/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<25
3basemodel/batch_normalization/AssignMovingAvg/decayў
<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp
1basemodel/batch_normalization/AssignMovingAvg/subSubDbasemodel/batch_normalization/AssignMovingAvg/ReadVariableOp:value:06basemodel/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization/AssignMovingAvg/sub
1basemodel/batch_normalization/AssignMovingAvg/mulMul5basemodel/batch_normalization/AssignMovingAvg/sub:z:0<basemodel/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization/AssignMovingAvg/mulе
-basemodel/batch_normalization/AssignMovingAvgAssignSubVariableOpEbasemodel_batch_normalization_assignmovingavg_readvariableop_resource5basemodel/batch_normalization/AssignMovingAvg/mul:z:0=^basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02/
-basemodel/batch_normalization/AssignMovingAvgГ
5basemodel/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<27
5basemodel/batch_normalization/AssignMovingAvg_1/decay
>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp
3basemodel/batch_normalization/AssignMovingAvg_1/subSubFbasemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:08basemodel/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_1/sub
3basemodel/batch_normalization/AssignMovingAvg_1/mulMul7basemodel/batch_normalization/AssignMovingAvg_1/sub:z:0>basemodel/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization/AssignMovingAvg_1/mulп
/basemodel/batch_normalization/AssignMovingAvg_1AssignSubVariableOpGbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource7basemodel/batch_normalization/AssignMovingAvg_1/mul:z:0?^basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization/AssignMovingAvg_1Ѓ
-basemodel/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-basemodel/batch_normalization/batchnorm/add/yњ
+basemodel/batch_normalization/batchnorm/addAddV28basemodel/batch_normalization/moments/Squeeze_1:output:06basemodel/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/addН
-basemodel/batch_normalization/batchnorm/RsqrtRsqrt/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/Rsqrtј
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp§
+basemodel/batch_normalization/batchnorm/mulMul1basemodel/batch_normalization/batchnorm/Rsqrt:y:0Bbasemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/mulљ
-basemodel/batch_normalization/batchnorm/mul_1Mul*basemodel/stream_0_conv_1/BiasAdd:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2/
-basemodel/batch_normalization/batchnorm/mul_1ѓ
-basemodel/batch_normalization/batchnorm/mul_2Mul6basemodel/batch_normalization/moments/Squeeze:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/mul_2ь
6basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype028
6basemodel/batch_normalization/batchnorm/ReadVariableOpљ
+basemodel/batch_normalization/batchnorm/subSub>basemodel/batch_normalization/batchnorm/ReadVariableOp:value:01basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/sub
-basemodel/batch_normalization/batchnorm/add_1AddV21basemodel/batch_normalization/batchnorm/mul_1:z:0/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2/
-basemodel/batch_normalization/batchnorm/add_1Ў
basemodel/activation_2/TanhTanh3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
basemodel/activation_2/TanhЎ
basemodel/activation_1/TanhTanh3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
basemodel/activation_1/TanhЈ
basemodel/activation/TanhTanh1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
basemodel/activation/Tanh
+basemodel/stream_2_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_2_maxpool_1/ExpandDims/dimђ
'basemodel/stream_2_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation_2/Tanh:y:04basemodel/stream_2_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@2)
'basemodel/stream_2_maxpool_1/ExpandDimsі
$basemodel/stream_2_maxpool_1/MaxPoolMaxPool0basemodel/stream_2_maxpool_1/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_2_maxpool_1/MaxPoolг
$basemodel/stream_2_maxpool_1/SqueezeSqueeze-basemodel/stream_2_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2&
$basemodel/stream_2_maxpool_1/Squeeze
+basemodel/stream_1_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_1_maxpool_1/ExpandDims/dimђ
'basemodel/stream_1_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation_1/Tanh:y:04basemodel/stream_1_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@2)
'basemodel/stream_1_maxpool_1/ExpandDimsі
$basemodel/stream_1_maxpool_1/MaxPoolMaxPool0basemodel/stream_1_maxpool_1/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_1_maxpool_1/MaxPoolг
$basemodel/stream_1_maxpool_1/SqueezeSqueeze-basemodel/stream_1_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2&
$basemodel/stream_1_maxpool_1/Squeeze
+basemodel/stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_0_maxpool_1/ExpandDims/dim№
'basemodel/stream_0_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation/Tanh:y:04basemodel/stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@2)
'basemodel/stream_0_maxpool_1/ExpandDimsі
$basemodel/stream_0_maxpool_1/MaxPoolMaxPool0basemodel/stream_0_maxpool_1/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_0_maxpool_1/MaxPoolг
$basemodel/stream_0_maxpool_1/SqueezeSqueeze-basemodel/stream_0_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2&
$basemodel/stream_0_maxpool_1/Squeeze
'basemodel/stream_2_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2)
'basemodel/stream_2_drop_1/dropout/Constь
%basemodel/stream_2_drop_1/dropout/MulMul-basemodel/stream_2_maxpool_1/Squeeze:output:00basemodel/stream_2_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2'
%basemodel/stream_2_drop_1/dropout/MulЏ
'basemodel/stream_2_drop_1/dropout/ShapeShape-basemodel/stream_2_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2)
'basemodel/stream_2_drop_1/dropout/ShapeЁ
>basemodel/stream_2_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_2_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
dtype0*
seedЗ*
seed2К2@
>basemodel/stream_2_drop_1/dropout/random_uniform/RandomUniformЉ
0basemodel/stream_2_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >22
0basemodel/stream_2_drop_1/dropout/GreaterEqual/yЊ
.basemodel/stream_2_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_2_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_2_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@20
.basemodel/stream_2_drop_1/dropout/GreaterEqualб
&basemodel/stream_2_drop_1/dropout/CastCast2basemodel/stream_2_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ}@2(
&basemodel/stream_2_drop_1/dropout/Castц
'basemodel/stream_2_drop_1/dropout/Mul_1Mul)basemodel/stream_2_drop_1/dropout/Mul:z:0*basemodel/stream_2_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2)
'basemodel/stream_2_drop_1/dropout/Mul_1
'basemodel/stream_1_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2)
'basemodel/stream_1_drop_1/dropout/Constь
%basemodel/stream_1_drop_1/dropout/MulMul-basemodel/stream_1_maxpool_1/Squeeze:output:00basemodel/stream_1_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2'
%basemodel/stream_1_drop_1/dropout/MulЏ
'basemodel/stream_1_drop_1/dropout/ShapeShape-basemodel/stream_1_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2)
'basemodel/stream_1_drop_1/dropout/ShapeЁ
>basemodel/stream_1_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_1_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
dtype0*
seedЗ*
seed2Й2@
>basemodel/stream_1_drop_1/dropout/random_uniform/RandomUniformЉ
0basemodel/stream_1_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >22
0basemodel/stream_1_drop_1/dropout/GreaterEqual/yЊ
.basemodel/stream_1_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_1_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_1_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@20
.basemodel/stream_1_drop_1/dropout/GreaterEqualб
&basemodel/stream_1_drop_1/dropout/CastCast2basemodel/stream_1_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ}@2(
&basemodel/stream_1_drop_1/dropout/Castц
'basemodel/stream_1_drop_1/dropout/Mul_1Mul)basemodel/stream_1_drop_1/dropout/Mul:z:0*basemodel/stream_1_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2)
'basemodel/stream_1_drop_1/dropout/Mul_1
'basemodel/stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2)
'basemodel/stream_0_drop_1/dropout/Constь
%basemodel/stream_0_drop_1/dropout/MulMul-basemodel/stream_0_maxpool_1/Squeeze:output:00basemodel/stream_0_drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2'
%basemodel/stream_0_drop_1/dropout/MulЏ
'basemodel/stream_0_drop_1/dropout/ShapeShape-basemodel/stream_0_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_1/dropout/ShapeЁ
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
dtype0*
seedЗ*
seed2И2@
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniformЉ
0basemodel/stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >22
0basemodel/stream_0_drop_1/dropout/GreaterEqual/yЊ
.basemodel/stream_0_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_0_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@20
.basemodel/stream_0_drop_1/dropout/GreaterEqualб
&basemodel/stream_0_drop_1/dropout/CastCast2basemodel/stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ}@2(
&basemodel/stream_0_drop_1/dropout/Castц
'basemodel/stream_0_drop_1/dropout/Mul_1Mul)basemodel/stream_0_drop_1/dropout/Mul:z:0*basemodel/stream_0_drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2)
'basemodel/stream_0_drop_1/dropout/Mul_1И
9basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9basemodel/global_average_pooling1d/Mean/reduction_indices§
'basemodel/global_average_pooling1d/MeanMean+basemodel/stream_0_drop_1/dropout/Mul_1:z:0Bbasemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2)
'basemodel/global_average_pooling1d/MeanМ
;basemodel/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d_1/Mean/reduction_indices
)basemodel/global_average_pooling1d_1/MeanMean+basemodel/stream_1_drop_1/dropout/Mul_1:z:0Dbasemodel/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2+
)basemodel/global_average_pooling1d_1/MeanМ
;basemodel/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d_2/Mean/reduction_indices
)basemodel/global_average_pooling1d_2/MeanMean+basemodel/stream_2_drop_1/dropout/Mul_1:z:0Dbasemodel/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2+
)basemodel/global_average_pooling1d_2/Mean
!basemodel/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!basemodel/concatenate/concat/axisЪ
basemodel/concatenate/concatConcatV20basemodel/global_average_pooling1d/Mean:output:02basemodel/global_average_pooling1d_1/Mean:output:02basemodel/global_average_pooling1d_2/Mean:output:0*basemodel/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџР2
basemodel/concatenate/concatФ
'basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02)
'basemodel/dense_1/MatMul/ReadVariableOpШ
basemodel/dense_1/MatMulMatMul%basemodel/concatenate/concat:output:0/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
basemodel/dense_1/MatMulТ
(basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(basemodel/dense_1/BiasAdd/ReadVariableOpЩ
basemodel/dense_1/BiasAddBiasAdd"basemodel/dense_1/MatMul:product:00basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
basemodel/dense_1/BiasAddЪ
>basemodel/batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2@
>basemodel/batch_normalization_3/moments/mean/reduction_indices
,basemodel/batch_normalization_3/moments/meanMean"basemodel/dense_1/BiasAdd:output:0Gbasemodel/batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2.
,basemodel/batch_normalization_3/moments/meanм
4basemodel/batch_normalization_3/moments/StopGradientStopGradient5basemodel/batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:@26
4basemodel/batch_normalization_3/moments/StopGradient 
9basemodel/batch_normalization_3/moments/SquaredDifferenceSquaredDifference"basemodel/dense_1/BiasAdd:output:0=basemodel/batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2;
9basemodel/batch_normalization_3/moments/SquaredDifferenceв
Bbasemodel/batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bbasemodel/batch_normalization_3/moments/variance/reduction_indicesВ
0basemodel/batch_normalization_3/moments/varianceMean=basemodel/batch_normalization_3/moments/SquaredDifference:z:0Kbasemodel/batch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(22
0basemodel/batch_normalization_3/moments/varianceр
/basemodel/batch_normalization_3/moments/SqueezeSqueeze5basemodel/batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 21
/basemodel/batch_normalization_3/moments/Squeezeш
1basemodel/batch_normalization_3/moments/Squeeze_1Squeeze9basemodel/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 23
1basemodel/batch_normalization_3/moments/Squeeze_1Г
5basemodel/batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<27
5basemodel/batch_normalization_3/AssignMovingAvg/decay
>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02@
>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp
3basemodel/batch_normalization_3/AssignMovingAvg/subSubFbasemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:08basemodel/batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization_3/AssignMovingAvg/sub
3basemodel/batch_normalization_3/AssignMovingAvg/mulMul7basemodel/batch_normalization_3/AssignMovingAvg/sub:z:0>basemodel/batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@25
3basemodel/batch_normalization_3/AssignMovingAvg/mulп
/basemodel/batch_normalization_3/AssignMovingAvgAssignSubVariableOpGbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource7basemodel/batch_normalization_3/AssignMovingAvg/mul:z:0?^basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization_3/AssignMovingAvgЗ
7basemodel/batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<29
7basemodel/batch_normalization_3/AssignMovingAvg_1/decay
@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp 
5basemodel/batch_normalization_3/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_3/AssignMovingAvg_1/sub
5basemodel/batch_normalization_3/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_3/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_3/AssignMovingAvg_1/mulщ
1basemodel/batch_normalization_3/AssignMovingAvg_1AssignSubVariableOpIbasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_3/AssignMovingAvg_1/mul:z:0A^basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_3/AssignMovingAvg_1Ї
/basemodel/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:21
/basemodel/batch_normalization_3/batchnorm/add/y
-basemodel/batch_normalization_3/batchnorm/addAddV2:basemodel/batch_normalization_3/moments/Squeeze_1:output:08basemodel/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_3/batchnorm/addУ
/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_3/batchnorm/Rsqrtў
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp
-basemodel/batch_normalization_3/batchnorm/mulMul3basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_3/batchnorm/mulђ
/basemodel/batch_normalization_3/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@21
/basemodel/batch_normalization_3/batchnorm/mul_1ћ
/basemodel/batch_normalization_3/batchnorm/mul_2Mul8basemodel/batch_normalization_3/moments/Squeeze:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_3/batchnorm/mul_2ђ
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_3/batchnorm/ReadVariableOp
-basemodel/batch_normalization_3/batchnorm/subSub@basemodel/batch_normalization_3/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_3/batchnorm/sub
/basemodel/batch_normalization_3/batchnorm/add_1AddV23basemodel/batch_normalization_3/batchnorm/mul_1:z:01basemodel/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@21
/basemodel/batch_normalization_3/batchnorm/add_1
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constј
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs­
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1й
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_0_conv_1/kernel/Regularizer/mul/xм
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulй
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addў
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square­
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2р
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1и
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Constј
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs­
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1й
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_1_conv_1/kernel/Regularizer/mul/xм
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulй
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addў
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square­
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2р
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1и
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Constј
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs­
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1й
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_2_conv_1/kernel/Regularizer/mul/xм
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulй
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addў
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square­
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2р
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1и
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Constа
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpЈ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2 
dense_1/kernel/Regularizer/Abs
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1Й
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_1/kernel/Regularizer/mul/xМ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addж
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpД
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2#
!dense_1/kernel/Regularizer/Square
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2Р
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"dense_1/kernel/Regularizer/mul_1/xФ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1И
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1
IdentityIdentity3basemodel/batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

IdentityЙ
NoOpNoOp.^basemodel/batch_normalization/AssignMovingAvg=^basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_1?^basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_1/AssignMovingAvg?^basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_1A^basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_2/AssignMovingAvg?^basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_1A^basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_3/AssignMovingAvg?^basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_3/AssignMovingAvg_1A^basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_3/batchnorm/ReadVariableOp=^basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 2^
-basemodel/batch_normalization/AssignMovingAvg-basemodel/batch_normalization/AssignMovingAvg2|
<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp2b
/basemodel/batch_normalization/AssignMovingAvg_1/basemodel/batch_normalization/AssignMovingAvg_12
>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp2p
6basemodel/batch_normalization/batchnorm/ReadVariableOp6basemodel/batch_normalization/batchnorm/ReadVariableOp2x
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp2b
/basemodel/batch_normalization_1/AssignMovingAvg/basemodel/batch_normalization_1/AssignMovingAvg2
>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp2f
1basemodel/batch_normalization_1/AssignMovingAvg_11basemodel/batch_normalization_1/AssignMovingAvg_12
@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2t
8basemodel/batch_normalization_1/batchnorm/ReadVariableOp8basemodel/batch_normalization_1/batchnorm/ReadVariableOp2|
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp2b
/basemodel/batch_normalization_2/AssignMovingAvg/basemodel/batch_normalization_2/AssignMovingAvg2
>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp2f
1basemodel/batch_normalization_2/AssignMovingAvg_11basemodel/batch_normalization_2/AssignMovingAvg_12
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2t
8basemodel/batch_normalization_2/batchnorm/ReadVariableOp8basemodel/batch_normalization_2/batchnorm/ReadVariableOp2|
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp2b
/basemodel/batch_normalization_3/AssignMovingAvg/basemodel/batch_normalization_3/AssignMovingAvg2
>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp2f
1basemodel/batch_normalization_3/AssignMovingAvg_11basemodel/batch_normalization_3/AssignMovingAvg_12
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
п
M
1__inference_stream_1_drop_1_layer_call_fn_3599951

inputs
identityб
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_35958032
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs

Џ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3599388

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

IdentityТ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџњ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
э
а
5__inference_batch_normalization_layer_call_fn_3599461

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_35957332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџњ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
И
Б
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3599494

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2

IdentityТ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Й+
ы
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3594958

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradientБ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
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
з#<2
AssignMovingAvg/decayЄ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulП
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
з#<2
AssignMovingAvg_1/decayЊ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mulЩ
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2

Identityђ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
И
Б
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3594898

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2

IdentityТ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
ё
в
7__inference_batch_normalization_2_layer_call_fn_3599781

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_35956752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџњ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
Ќ
k
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_3595780

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@2

ExpandDims
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2	
MaxPool|
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ@:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs

j
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_3599907

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
+
ы
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3599582

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradientЉ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
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
з#<2
AssignMovingAvg/decayЄ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulП
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
з#<2
AssignMovingAvg_1/decayЊ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mulЩ
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityђ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџњ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
њ,

L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_3599251

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂ"conv1d/ExpandDims_1/ReadVariableOpЂ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ2
conv1d/ExpandDimsИ
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
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2	
BiasAdd
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Constо
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs­
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1й
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_1_conv_1/kernel/Regularizer/mul/xм
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulй
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addф
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square­
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2р
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1и
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1p
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityџ
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџњ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs

Б
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3599708

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

IdentityТ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџњ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
ј
љ
__inference_loss_fn_2_3600281T
>stream_2_conv_1_kernel_regularizer_abs_readvariableop_resource:@
identityЂ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Constё
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_2_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs­
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1й
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_2_conv_1/kernel/Regularizer/mul/xм
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulй
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addї
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>stream_2_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square­
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2р
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1и
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1v
IdentityIdentity,stream_2_conv_1/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityС
NoOpNoOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp
њ
o
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_3599115

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeд
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ*
dtype0*
seedЗ*
seed2И2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yУ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџњ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџњ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
З+
щ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3599368

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradientБ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
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
з#<2
AssignMovingAvg/decayЄ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulП
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
з#<2
AssignMovingAvg_1/decayЊ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mulЩ
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2

Identityђ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Ё
о
__inference_loss_fn_3_3600301I
6dense_1_kernel_regularizer_abs_readvariableop_resource:	Р@
identityЂ-dense_1/kernel/Regularizer/Abs/ReadVariableOpЂ0dense_1/kernel/Regularizer/Square/ReadVariableOp
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Constж
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	Р@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpЈ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2 
dense_1/kernel/Regularizer/Abs
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1Й
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_1/kernel/Regularizer/mul/xМ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addм
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	Р@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpД
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2#
!dense_1/kernel/Regularizer/Square
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2Р
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"dense_1/kernel/Regularizer/mul_1/xФ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1И
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1n
IdentityIdentity$dense_1/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityБ
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
њi
Л
#__inference__traced_restore_3600478
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
"assignvariableop_12_dense_1_kernel:	Р@.
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
identity_25ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*	
value	B	B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesР
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЈ
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

IdentityІ
AssignVariableOpAssignVariableOp'assignvariableop_stream_0_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ќ
AssignVariableOp_1AssignVariableOp'assignvariableop_1_stream_0_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ў
AssignVariableOp_2AssignVariableOp)assignvariableop_2_stream_1_conv_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ќ
AssignVariableOp_3AssignVariableOp'assignvariableop_3_stream_1_conv_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ў
AssignVariableOp_4AssignVariableOp)assignvariableop_4_stream_2_conv_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ќ
AssignVariableOp_5AssignVariableOp'assignvariableop_5_stream_2_conv_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Б
AssignVariableOp_6AssignVariableOp,assignvariableop_6_batch_normalization_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7А
AssignVariableOp_7AssignVariableOp+assignvariableop_7_batch_normalization_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Г
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9В
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10З
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_2_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ж
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_2_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Њ
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ј
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14З
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_3_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ж
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_3_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Л
AssignVariableOp_16AssignVariableOp3assignvariableop_16_batch_normalization_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17П
AssignVariableOp_17AssignVariableOp7assignvariableop_17_batch_normalization_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Н
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_1_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19С
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_1_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Н
AssignVariableOp_20AssignVariableOp5assignvariableop_20_batch_normalization_2_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21С
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_normalization_2_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Н
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23С
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
Identity_25ж
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
Н
s
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_3595316

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
:џџџџџџџџџџџџџџџџџџ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	
а
5__inference_batch_normalization_layer_call_fn_3599448

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_35947962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs

s
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_3600039

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
:џџџџџџџџџ@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
Ў
P
4__inference_stream_0_maxpool_1_layer_call_fn_3599845

inputs
identityц
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_35952102
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3599995

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
:џџџџџџџџџ@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
О

+__inference_basemodel_layer_call_fn_3596008
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

unknown_17:	Р@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityЂStatefulPartitionedCallМ
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
:џџџџџџџџџ@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_35959572
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesz
x:џџџџџџџџџњ:џџџџџџџџџњ:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs_2
§!
й
D__inference_dense_1_layer_call_and_return_conditional_losses_3600123

inputs1
matmul_readvariableop_resource:	Р@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ-dense_1/kernel/Regularizer/Abs/ReadVariableOpЂ0dense_1/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAdd
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/ConstО
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpЈ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2 
dense_1/kernel/Regularizer/Abs
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1Й
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_1/kernel/Regularizer/mul/xМ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addФ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpД
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2#
!dense_1/kernel/Regularizer/Square
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2Р
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"dense_1/kernel/Regularizer/mul_1/xФ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1И
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityт
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџР: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
я
в
7__inference_batch_normalization_2_layer_call_fn_3599794

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_35963332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџњ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
Ќ
k
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_3599866

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@2

ExpandDims
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2	
MaxPool|
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ@:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs

Ђ
1__inference_stream_2_conv_1_layer_call_fn_3599314

inputs
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_35955782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџњ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
Э
g
-__inference_concatenate_layer_call_fn_3600064
inputs_0
inputs_1
inputs_2
identityт
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_35958412
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ@:џџџџџџџџџ@:џџџџџџџџџ@:Q M
'
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
inputs/2
Њ:

 __inference__traced_save_3600396
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

identity_1ЂMergeV2Checkpoints
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
Const_1
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*	
value	B	B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesК
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_stream_0_conv_1_kernel_read_readvariableop/savev2_stream_0_conv_1_bias_read_readvariableop1savev2_stream_1_conv_1_kernel_read_readvariableop/savev2_stream_1_conv_1_bias_read_readvariableop1savev2_stream_2_conv_1_kernel_read_readvariableop/savev2_stream_2_conv_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
22
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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

identity_1Identity_1:output:0*Ц
_input_shapesД
Б: :@:@:@:@:@:@:@:@:@:@:@:@:	Р@:@:@:@:@:@:@:@:@:@:@:@: 2(
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
:	Р@: 
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
Н
s
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_3595340

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
:џџџџџџџџџџџџџџџџџџ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

j
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_3595803

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
С
j
1__inference_stream_2_drop_1_layer_call_fn_3599983

inputs
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_35961282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
і
Б
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3595378

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

IdentityТ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
яЁ
й
"__inference__wrapped_model_3594712
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
8model_1_basemodel_dense_1_matmul_readvariableop_resource:	Р@G
9model_1_basemodel_dense_1_biasadd_readvariableop_resource:@W
Imodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_resource:@[
Mmodel_1_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:@Y
Kmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource:@Y
Kmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource:@
identityЂ>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOpЂ@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1Ђ@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2ЂBmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpЂ@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpЂBmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ЂBmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ЂDmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpЂ@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOpЂBmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ЂBmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ЂDmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpЂ@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOpЂBmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1ЂBmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2ЂDmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpЂ0model_1/basemodel/dense_1/BiasAdd/ReadVariableOpЂ/model_1/basemodel/dense_1/MatMul/ReadVariableOpЂ8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpЂDmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpЂ8model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpЂDmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpЂ8model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpЂDmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpА
.model_1/basemodel/stream_2_input_drop/IdentityIdentityleft_inputs*
T0*,
_output_shapes
:џџџџџџџџџњ20
.model_1/basemodel/stream_2_input_drop/IdentityА
.model_1/basemodel/stream_1_input_drop/IdentityIdentityleft_inputs*
T0*,
_output_shapes
:џџџџџџџџџњ20
.model_1/basemodel/stream_1_input_drop/IdentityА
.model_1/basemodel/stream_0_input_drop/IdentityIdentityleft_inputs*
T0*,
_output_shapes
:џџџџџџџџџњ20
.model_1/basemodel/stream_0_input_drop/IdentityН
7model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ29
7model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimЎ
3model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims
ExpandDims7model_1/basemodel/stream_2_input_drop/Identity:output:0@model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ25
3model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims
Dmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpMmodel_1_basemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpИ
9model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dimП
5model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1
ExpandDimsLmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Bmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@27
5model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1П
(model_1/basemodel/stream_2_conv_1/conv1dConv2D<model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims:output:0>model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2*
(model_1/basemodel/stream_2_conv_1/conv1dљ
0model_1/basemodel/stream_2_conv_1/conv1d/SqueezeSqueeze1model_1/basemodel/stream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ22
0model_1/basemodel/stream_2_conv_1/conv1d/Squeezeђ
8model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOpAmodel_1_basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp
)model_1/basemodel/stream_2_conv_1/BiasAddBiasAdd9model_1/basemodel/stream_2_conv_1/conv1d/Squeeze:output:0@model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2+
)model_1/basemodel/stream_2_conv_1/BiasAddН
7model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ29
7model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimЎ
3model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims
ExpandDims7model_1/basemodel/stream_1_input_drop/Identity:output:0@model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ25
3model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims
Dmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpMmodel_1_basemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpИ
9model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dimП
5model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1
ExpandDimsLmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Bmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@27
5model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1П
(model_1/basemodel/stream_1_conv_1/conv1dConv2D<model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims:output:0>model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2*
(model_1/basemodel/stream_1_conv_1/conv1dљ
0model_1/basemodel/stream_1_conv_1/conv1d/SqueezeSqueeze1model_1/basemodel/stream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ22
0model_1/basemodel/stream_1_conv_1/conv1d/Squeezeђ
8model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOpAmodel_1_basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp
)model_1/basemodel/stream_1_conv_1/BiasAddBiasAdd9model_1/basemodel/stream_1_conv_1/conv1d/Squeeze:output:0@model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2+
)model_1/basemodel/stream_1_conv_1/BiasAddН
7model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ29
7model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimЎ
3model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims7model_1/basemodel/stream_0_input_drop/Identity:output:0@model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ25
3model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims
Dmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpMmodel_1_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpИ
9model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimП
5model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1
ExpandDimsLmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Bmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@27
5model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1П
(model_1/basemodel/stream_0_conv_1/conv1dConv2D<model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims:output:0>model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2*
(model_1/basemodel/stream_0_conv_1/conv1dљ
0model_1/basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze1model_1/basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ22
0model_1/basemodel/stream_0_conv_1/conv1d/Squeezeђ
8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOpAmodel_1_basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp
)model_1/basemodel/stream_0_conv_1/BiasAddBiasAdd9model_1/basemodel/stream_0_conv_1/conv1d/Squeeze:output:0@model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2+
)model_1/basemodel/stream_0_conv_1/BiasAdd
@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpImodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02B
@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOpЗ
7model_1/basemodel/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:29
7model_1/basemodel/batch_normalization_2/batchnorm/add/yЈ
5model_1/basemodel/batch_normalization_2/batchnorm/addAddV2Hmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:0@model_1/basemodel/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_2/batchnorm/addл
7model_1/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt9model_1/basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@29
7model_1/basemodel/batch_normalization_2/batchnorm/Rsqrt
Dmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpMmodel_1_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpЅ
5model_1/basemodel/batch_normalization_2/batchnorm/mulMul;model_1/basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Lmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_2/batchnorm/mul
7model_1/basemodel/batch_normalization_2/batchnorm/mul_1Mul2model_1/basemodel/stream_2_conv_1/BiasAdd:output:09model_1/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@29
7model_1/basemodel/batch_normalization_2/batchnorm/mul_1
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpKmodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1Ѕ
7model_1/basemodel/batch_normalization_2/batchnorm/mul_2MulJmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:value:09model_1/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@29
7model_1/basemodel/batch_normalization_2/batchnorm/mul_2
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpKmodel_1_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2Ѓ
5model_1/basemodel/batch_normalization_2/batchnorm/subSubJmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0;model_1/basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_2/batchnorm/subЊ
7model_1/basemodel/batch_normalization_2/batchnorm/add_1AddV2;model_1/basemodel/batch_normalization_2/batchnorm/mul_1:z:09model_1/basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@29
7model_1/basemodel/batch_normalization_2/batchnorm/add_1
@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpImodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02B
@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpЗ
7model_1/basemodel/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:29
7model_1/basemodel/batch_normalization_1/batchnorm/add/yЈ
5model_1/basemodel/batch_normalization_1/batchnorm/addAddV2Hmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:0@model_1/basemodel/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_1/batchnorm/addл
7model_1/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt9model_1/basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@29
7model_1/basemodel/batch_normalization_1/batchnorm/Rsqrt
Dmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpMmodel_1_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpЅ
5model_1/basemodel/batch_normalization_1/batchnorm/mulMul;model_1/basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Lmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_1/batchnorm/mul
7model_1/basemodel/batch_normalization_1/batchnorm/mul_1Mul2model_1/basemodel/stream_1_conv_1/BiasAdd:output:09model_1/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@29
7model_1/basemodel/batch_normalization_1/batchnorm/mul_1
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpKmodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1Ѕ
7model_1/basemodel/batch_normalization_1/batchnorm/mul_2MulJmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1:value:09model_1/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@29
7model_1/basemodel/batch_normalization_1/batchnorm/mul_2
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpKmodel_1_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2Ѓ
5model_1/basemodel/batch_normalization_1/batchnorm/subSubJmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0;model_1/basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_1/batchnorm/subЊ
7model_1/basemodel/batch_normalization_1/batchnorm/add_1AddV2;model_1/basemodel/batch_normalization_1/batchnorm/mul_1:z:09model_1/basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@29
7model_1/basemodel/batch_normalization_1/batchnorm/add_1
>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOpGmodel_1_basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOpГ
5model_1/basemodel/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:27
5model_1/basemodel/batch_normalization/batchnorm/add/y 
3model_1/basemodel/batch_normalization/batchnorm/addAddV2Fmodel_1/basemodel/batch_normalization/batchnorm/ReadVariableOp:value:0>model_1/basemodel/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@25
3model_1/basemodel/batch_normalization/batchnorm/addе
5model_1/basemodel/batch_normalization/batchnorm/RsqrtRsqrt7model_1/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization/batchnorm/Rsqrt
Bmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpKmodel_1_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp
3model_1/basemodel/batch_normalization/batchnorm/mulMul9model_1/basemodel/batch_normalization/batchnorm/Rsqrt:y:0Jmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@25
3model_1/basemodel/batch_normalization/batchnorm/mul
5model_1/basemodel/batch_normalization/batchnorm/mul_1Mul2model_1/basemodel/stream_0_conv_1/BiasAdd:output:07model_1/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@27
5model_1/basemodel/batch_normalization/batchnorm/mul_1
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpImodel_1_basemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1
5model_1/basemodel/batch_normalization/batchnorm/mul_2MulHmodel_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1:value:07model_1/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization/batchnorm/mul_2
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpImodel_1_basemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02B
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2
3model_1/basemodel/batch_normalization/batchnorm/subSubHmodel_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2:value:09model_1/basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@25
3model_1/basemodel/batch_normalization/batchnorm/subЂ
5model_1/basemodel/batch_normalization/batchnorm/add_1AddV29model_1/basemodel/batch_normalization/batchnorm/mul_1:z:07model_1/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@27
5model_1/basemodel/batch_normalization/batchnorm/add_1Ц
#model_1/basemodel/activation_2/TanhTanh;model_1/basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2%
#model_1/basemodel/activation_2/TanhЦ
#model_1/basemodel/activation_1/TanhTanh;model_1/basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2%
#model_1/basemodel/activation_1/TanhР
!model_1/basemodel/activation/TanhTanh9model_1/basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2#
!model_1/basemodel/activation/TanhЌ
3model_1/basemodel/stream_2_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :25
3model_1/basemodel/stream_2_maxpool_1/ExpandDims/dim
/model_1/basemodel/stream_2_maxpool_1/ExpandDims
ExpandDims'model_1/basemodel/activation_2/Tanh:y:0<model_1/basemodel/stream_2_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@21
/model_1/basemodel/stream_2_maxpool_1/ExpandDims
,model_1/basemodel/stream_2_maxpool_1/MaxPoolMaxPool8model_1/basemodel/stream_2_maxpool_1/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2.
,model_1/basemodel/stream_2_maxpool_1/MaxPoolы
,model_1/basemodel/stream_2_maxpool_1/SqueezeSqueeze5model_1/basemodel/stream_2_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2.
,model_1/basemodel/stream_2_maxpool_1/SqueezeЌ
3model_1/basemodel/stream_1_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :25
3model_1/basemodel/stream_1_maxpool_1/ExpandDims/dim
/model_1/basemodel/stream_1_maxpool_1/ExpandDims
ExpandDims'model_1/basemodel/activation_1/Tanh:y:0<model_1/basemodel/stream_1_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@21
/model_1/basemodel/stream_1_maxpool_1/ExpandDims
,model_1/basemodel/stream_1_maxpool_1/MaxPoolMaxPool8model_1/basemodel/stream_1_maxpool_1/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2.
,model_1/basemodel/stream_1_maxpool_1/MaxPoolы
,model_1/basemodel/stream_1_maxpool_1/SqueezeSqueeze5model_1/basemodel/stream_1_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2.
,model_1/basemodel/stream_1_maxpool_1/SqueezeЌ
3model_1/basemodel/stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :25
3model_1/basemodel/stream_0_maxpool_1/ExpandDims/dim
/model_1/basemodel/stream_0_maxpool_1/ExpandDims
ExpandDims%model_1/basemodel/activation/Tanh:y:0<model_1/basemodel/stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@21
/model_1/basemodel/stream_0_maxpool_1/ExpandDims
,model_1/basemodel/stream_0_maxpool_1/MaxPoolMaxPool8model_1/basemodel/stream_0_maxpool_1/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2.
,model_1/basemodel/stream_0_maxpool_1/MaxPoolы
,model_1/basemodel/stream_0_maxpool_1/SqueezeSqueeze5model_1/basemodel/stream_0_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2.
,model_1/basemodel/stream_0_maxpool_1/Squeezeб
*model_1/basemodel/stream_2_drop_1/IdentityIdentity5model_1/basemodel/stream_2_maxpool_1/Squeeze:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2,
*model_1/basemodel/stream_2_drop_1/Identityб
*model_1/basemodel/stream_1_drop_1/IdentityIdentity5model_1/basemodel/stream_1_maxpool_1/Squeeze:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2,
*model_1/basemodel/stream_1_drop_1/Identityб
*model_1/basemodel/stream_0_drop_1/IdentityIdentity5model_1/basemodel/stream_0_maxpool_1/Squeeze:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2,
*model_1/basemodel/stream_0_drop_1/IdentityШ
Amodel_1/basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Amodel_1/basemodel/global_average_pooling1d/Mean/reduction_indices
/model_1/basemodel/global_average_pooling1d/MeanMean3model_1/basemodel/stream_0_drop_1/Identity:output:0Jmodel_1/basemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@21
/model_1/basemodel/global_average_pooling1d/MeanЬ
Cmodel_1/basemodel/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2E
Cmodel_1/basemodel/global_average_pooling1d_1/Mean/reduction_indicesЃ
1model_1/basemodel/global_average_pooling1d_1/MeanMean3model_1/basemodel/stream_1_drop_1/Identity:output:0Lmodel_1/basemodel/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@23
1model_1/basemodel/global_average_pooling1d_1/MeanЬ
Cmodel_1/basemodel/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2E
Cmodel_1/basemodel/global_average_pooling1d_2/Mean/reduction_indicesЃ
1model_1/basemodel/global_average_pooling1d_2/MeanMean3model_1/basemodel/stream_2_drop_1/Identity:output:0Lmodel_1/basemodel/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@23
1model_1/basemodel/global_average_pooling1d_2/Mean
)model_1/basemodel/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_1/basemodel/concatenate/concat/axisњ
$model_1/basemodel/concatenate/concatConcatV28model_1/basemodel/global_average_pooling1d/Mean:output:0:model_1/basemodel/global_average_pooling1d_1/Mean:output:0:model_1/basemodel/global_average_pooling1d_2/Mean:output:02model_1/basemodel/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџР2&
$model_1/basemodel/concatenate/concatЦ
*model_1/basemodel/dense_1_dropout/IdentityIdentity-model_1/basemodel/concatenate/concat:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2,
*model_1/basemodel/dense_1_dropout/Identityм
/model_1/basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp8model_1_basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype021
/model_1/basemodel/dense_1/MatMul/ReadVariableOpю
 model_1/basemodel/dense_1/MatMulMatMul3model_1/basemodel/dense_1_dropout/Identity:output:07model_1/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2"
 model_1/basemodel/dense_1/MatMulк
0model_1/basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp9model_1_basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0model_1/basemodel/dense_1/BiasAdd/ReadVariableOpщ
!model_1/basemodel/dense_1/BiasAddBiasAdd*model_1/basemodel/dense_1/MatMul:product:08model_1/basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2#
!model_1/basemodel/dense_1/BiasAdd
@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpImodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02B
@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOpЗ
7model_1/basemodel/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:29
7model_1/basemodel/batch_normalization_3/batchnorm/add/yЈ
5model_1/basemodel/batch_normalization_3/batchnorm/addAddV2Hmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp:value:0@model_1/basemodel/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_3/batchnorm/addл
7model_1/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt9model_1/basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@29
7model_1/basemodel/batch_normalization_3/batchnorm/Rsqrt
Dmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpMmodel_1_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpЅ
5model_1/basemodel/batch_normalization_3/batchnorm/mulMul;model_1/basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Lmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_3/batchnorm/mul
7model_1/basemodel/batch_normalization_3/batchnorm/mul_1Mul*model_1/basemodel/dense_1/BiasAdd:output:09model_1/basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@29
7model_1/basemodel/batch_normalization_3/batchnorm/mul_1
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpKmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1Ѕ
7model_1/basemodel/batch_normalization_3/batchnorm/mul_2MulJmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:value:09model_1/basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@29
7model_1/basemodel/batch_normalization_3/batchnorm/mul_2
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpKmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2Ѓ
5model_1/basemodel/batch_normalization_3/batchnorm/subSubJmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2:value:0;model_1/basemodel/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_3/batchnorm/subЅ
7model_1/basemodel/batch_normalization_3/batchnorm/add_1AddV2;model_1/basemodel/batch_normalization_3/batchnorm/mul_1:z:09model_1/basemodel/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@29
7model_1/basemodel/batch_normalization_3/batchnorm/add_1
IdentityIdentity;model_1/basemodel/batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity
NoOpNoOp?^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOpA^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1A^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2C^model_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp1^model_1/basemodel/dense_1/BiasAdd/ReadVariableOp0^model_1/basemodel/dense_1/MatMul/ReadVariableOp9^model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp9^model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp9^model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 2
>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp2
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_12
@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_22
Bmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpBmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp2
@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp2
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_12
Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2Bmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_22
Dmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpDmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp2
@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp2
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_12
Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2Bmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_22
Dmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpDmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp2
@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp2
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_12
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_22
Dmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpDmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp2d
0model_1/basemodel/dense_1/BiasAdd/ReadVariableOp0model_1/basemodel/dense_1/BiasAdd/ReadVariableOp2b
/model_1/basemodel/dense_1/MatMul/ReadVariableOp/model_1/basemodel/dense_1/MatMul/ReadVariableOp2t
8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2
Dmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpDmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2t
8model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp8model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp2
Dmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpDmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp2t
8model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp8model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp2
Dmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpDmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:Y U
,
_output_shapes
:џџџџџџџџџњ
%
_user_specified_nameleft_inputs
щ
c
G__inference_activation_layer_call_and_return_conditional_losses_3595762

inputs
identityS
TanhTanhinputs*
T0*,
_output_shapes
:џџџџџџџџџњ@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ@:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
сш
А
F__inference_basemodel_layer_call_and_return_conditional_losses_3596642

inputs
inputs_1
inputs_2-
stream_2_conv_1_3596510:@%
stream_2_conv_1_3596512:@-
stream_1_conv_1_3596515:@%
stream_1_conv_1_3596517:@-
stream_0_conv_1_3596520:@%
stream_0_conv_1_3596522:@+
batch_normalization_2_3596525:@+
batch_normalization_2_3596527:@+
batch_normalization_2_3596529:@+
batch_normalization_2_3596531:@+
batch_normalization_1_3596534:@+
batch_normalization_1_3596536:@+
batch_normalization_1_3596538:@+
batch_normalization_1_3596540:@)
batch_normalization_3596543:@)
batch_normalization_3596545:@)
batch_normalization_3596547:@)
batch_normalization_3596549:@"
dense_1_3596566:	Р@
dense_1_3596568:@+
batch_normalization_3_3596571:@+
batch_normalization_3_3596573:@+
batch_normalization_3_3596575:@+
batch_normalization_3_3596577:@
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂ-dense_1/kernel/Regularizer/Abs/ReadVariableOpЂ0dense_1/kernel/Regularizer/Square/ReadVariableOpЂ'stream_0_conv_1/StatefulPartitionedCallЂ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ'stream_0_drop_1/StatefulPartitionedCallЂ+stream_0_input_drop/StatefulPartitionedCallЂ'stream_1_conv_1/StatefulPartitionedCallЂ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ'stream_1_drop_1/StatefulPartitionedCallЂ+stream_1_input_drop/StatefulPartitionedCallЂ'stream_2_conv_1/StatefulPartitionedCallЂ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ'stream_2_drop_1/StatefulPartitionedCallЂ+stream_2_input_drop/StatefulPartitionedCall
+stream_2_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_35964402-
+stream_2_input_drop/StatefulPartitionedCallЦ
+stream_1_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_1,^stream_2_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_35964172-
+stream_1_input_drop/StatefulPartitionedCallФ
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs,^stream_1_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_35963942-
+stream_0_input_drop/StatefulPartitionedCall№
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_2_input_drop/StatefulPartitionedCall:output:0stream_2_conv_1_3596510stream_2_conv_1_3596512*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_35955782)
'stream_2_conv_1/StatefulPartitionedCall№
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_1_input_drop/StatefulPartitionedCall:output:0stream_1_conv_1_3596515stream_1_conv_1_3596517*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_35956142)
'stream_1_conv_1/StatefulPartitionedCall№
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_3596520stream_0_conv_1_3596522*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_35956502)
'stream_0_conv_1/StatefulPartitionedCallЪ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_3596525batch_normalization_2_3596527batch_normalization_2_3596529batch_normalization_2_3596531*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_35963332/
-batch_normalization_2/StatefulPartitionedCallЪ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_3596534batch_normalization_1_3596536batch_normalization_1_3596538batch_normalization_1_3596540*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35962732/
-batch_normalization_1/StatefulPartitionedCallМ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_3596543batch_normalization_3596545batch_normalization_3596547batch_normalization_3596549*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_35962132-
+batch_normalization/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_35957482
activation_2/PartitionedCall
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_35957552
activation_1/PartitionedCall
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_35957622
activation/PartitionedCall
"stream_2_maxpool_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_35957712$
"stream_2_maxpool_1/PartitionedCall
"stream_1_maxpool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_35957802$
"stream_1_maxpool_1/PartitionedCall
"stream_0_maxpool_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_35957892$
"stream_0_maxpool_1/PartitionedCallм
'stream_2_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_2_maxpool_1/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_35961282)
'stream_2_drop_1/StatefulPartitionedCallи
'stream_1_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_1_maxpool_1/PartitionedCall:output:0(^stream_2_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_35961052)
'stream_1_drop_1/StatefulPartitionedCallи
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_0_maxpool_1/PartitionedCall:output:0(^stream_1_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_35960822)
'stream_0_drop_1/StatefulPartitionedCallВ
(global_average_pooling1d/PartitionedCallPartitionedCall0stream_0_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_35958172*
(global_average_pooling1d/PartitionedCallИ
*global_average_pooling1d_1/PartitionedCallPartitionedCall0stream_1_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_35958242,
*global_average_pooling1d_1/PartitionedCallИ
*global_average_pooling1d_2/PartitionedCallPartitionedCall0stream_2_drop_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_35958312,
*global_average_pooling1d_2/PartitionedCallљ
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:03global_average_pooling1d_1/PartitionedCall:output:03global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_35958412
concatenate/PartitionedCall
dense_1_dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_35960362!
dense_1_dropout/PartitionedCallЗ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_3596566dense_1_3596568*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_35958752!
dense_1/StatefulPartitionedCallН
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_3596571batch_normalization_3_3596573batch_normalization_3_3596575batch_normalization_3_3596577*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_35954382/
-batch_normalization_3/StatefulPartitionedCallІ
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_35958942$
"dense_activation_1/PartitionedCall
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/ConstЪ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_3596520*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs­
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1й
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_0_conv_1/kernel/Regularizer/mul/xм
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulй
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addа
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_1_3596520*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square­
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2р
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1и
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/ConstЪ
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_1_conv_1_3596515*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs­
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1й
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_1_conv_1/kernel/Regularizer/mul/xм
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulй
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addа
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_3596515*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square­
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2р
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1и
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/ConstЪ
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_3596510*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs­
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1й
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_2_conv_1/kernel/Regularizer/mul/xм
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulй
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addа
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_2_conv_1_3596510*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square­
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2р
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1и
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/ConstЏ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_3596566*
_output_shapes
:	Р@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpЈ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2 
dense_1/kernel/Regularizer/Abs
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1Й
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_1/kernel/Regularizer/mul/xМ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addЕ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_3596566*
_output_shapes
:	Р@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpД
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2#
!dense_1/kernel/Regularizer/Square
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2Р
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"dense_1/kernel/Regularizer/mul_1/xФ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1И
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity№
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall(^stream_1_conv_1/StatefulPartitionedCall6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_1_drop_1/StatefulPartitionedCall,^stream_1_input_drop/StatefulPartitionedCall(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_2_drop_1/StatefulPartitionedCall,^stream_2_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesz
x:џџџџџџџџџњ:џџџџџџџџџњ:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
'stream_0_drop_1/StatefulPartitionedCall'stream_0_drop_1/StatefulPartitionedCall2Z
+stream_0_input_drop/StatefulPartitionedCall+stream_0_input_drop/StatefulPartitionedCall2R
'stream_1_conv_1/StatefulPartitionedCall'stream_1_conv_1/StatefulPartitionedCall2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2R
'stream_1_drop_1/StatefulPartitionedCall'stream_1_drop_1/StatefulPartitionedCall2Z
+stream_1_input_drop/StatefulPartitionedCall+stream_1_input_drop/StatefulPartitionedCall2R
'stream_2_conv_1/StatefulPartitionedCall'stream_2_conv_1/StatefulPartitionedCall2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp2R
'stream_2_drop_1/StatefulPartitionedCall'stream_2_drop_1/StatefulPartitionedCall2Z
+stream_2_input_drop/StatefulPartitionedCall+stream_2_input_drop/StatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs:TP
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs:TP
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs

Б
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3595704

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

IdentityТ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџњ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
ы
Q
5__inference_stream_0_input_drop_layer_call_fn_3599093

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_35955462
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
§
j
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_3595848

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџР2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџР:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
+
ы
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3599742

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradientЉ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
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
з#<2
AssignMovingAvg/decayЄ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulП
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
з#<2
AssignMovingAvg_1/decayЊ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mulЩ
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityђ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџњ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
њ
o
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_3596394

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeд
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ*
dtype0*
seedЗ*
seed2З2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yУ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџњ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџњ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
+
щ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3596213

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradientЉ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
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
з#<2
AssignMovingAvg/decayЄ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulП
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
з#<2
AssignMovingAvg_1/decayЊ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mulЩ
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityђ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџњ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs

k
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_3595894

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ј
ч
)__inference_model_1_layer_call_fn_3597468
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

unknown_17:	Р@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityЂStatefulPartitionedCall
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
:џџџџџџџџџ@*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_35973642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:џџџџџџџџџњ
%
_user_specified_nameleft_inputs
њ,

L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_3599305

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂ"conv1d/ExpandDims_1/ReadVariableOpЂ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ2
conv1d/ExpandDimsИ
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
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2	
BiasAdd
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Constо
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs­
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1й
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_2_conv_1/kernel/Regularizer/mul/xм
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulй
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addф
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square­
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2р
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1и
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1p
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityџ
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџњ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs

Ђ
1__inference_stream_1_conv_1_layer_call_fn_3599260

inputs
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_35956142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџњ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs

j
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_3595796

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
њ
o
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_3596440

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeд
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ*
dtype0*
seedЗ*
seed2Й2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yУ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџњ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџњ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
ы
e
I__inference_activation_1_layer_call_and_return_conditional_losses_3595755

inputs
identityS
TanhTanhinputs*
T0*,
_output_shapes
:џџџџџџџџџњ@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ@:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
ю
k
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_3599919

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeг
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
dtype0*
seedЗ*
seed2И2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yТ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ}@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
ч
P
4__inference_stream_0_maxpool_1_layer_call_fn_3599850

inputs
identityд
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_35957892
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ@:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
њ,

L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_3595614

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂ"conv1d/ExpandDims_1/ReadVariableOpЂ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ2
conv1d/ExpandDimsИ
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
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2	
BiasAdd
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Constо
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs­
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1й
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_1_conv_1/kernel/Regularizer/mul/xм
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulй
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addф
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square­
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2р
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1и
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1p
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityџ
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџњ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
м
в
7__inference_batch_normalization_3_layer_call_fn_3600199

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_35953782
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
й
H
,__inference_activation_layer_call_fn_3599804

inputs
identityЭ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_35957622
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ@:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
ї

)__inference_dense_1_layer_call_fn_3600132

inputs
unknown:	Р@
	unknown_0:@
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_35958752
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџР: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
п
M
1__inference_stream_2_drop_1_layer_call_fn_3599978

inputs
identityб
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_35957962
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs

k
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_3599858

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

ExpandDimsБ
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

k
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_3595266

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

ExpandDimsБ
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
э
X
<__inference_global_average_pooling1d_2_layer_call_fn_3600049

inputs
identityи
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_35958312
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
г
M
1__inference_dense_1_dropout_layer_call_fn_3600078

inputs
identityЮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_35958482
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџР:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
З+
щ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3594796

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradientБ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
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
з#<2
AssignMovingAvg/decayЄ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulП
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
з#<2
AssignMovingAvg_1/decayЊ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mulЩ
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2

Identityђ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
{
§

D__inference_model_1_layer_call_and_return_conditional_losses_3597694
left_inputs'
basemodel_3597584:@
basemodel_3597586:@'
basemodel_3597588:@
basemodel_3597590:@'
basemodel_3597592:@
basemodel_3597594:@
basemodel_3597596:@
basemodel_3597598:@
basemodel_3597600:@
basemodel_3597602:@
basemodel_3597604:@
basemodel_3597606:@
basemodel_3597608:@
basemodel_3597610:@
basemodel_3597612:@
basemodel_3597614:@
basemodel_3597616:@
basemodel_3597618:@$
basemodel_3597620:	Р@
basemodel_3597622:@
basemodel_3597624:@
basemodel_3597626:@
basemodel_3597628:@
basemodel_3597630:@
identityЂ!basemodel/StatefulPartitionedCallЂ-dense_1/kernel/Regularizer/Abs/ReadVariableOpЂ0dense_1/kernel/Regularizer/Square/ReadVariableOpЂ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsleft_inputsleft_inputsbasemodel_3597584basemodel_3597586basemodel_3597588basemodel_3597590basemodel_3597592basemodel_3597594basemodel_3597596basemodel_3597598basemodel_3597600basemodel_3597602basemodel_3597604basemodel_3597606basemodel_3597608basemodel_3597610basemodel_3597612basemodel_3597614basemodel_3597616basemodel_3597618basemodel_3597620basemodel_3597622basemodel_3597624basemodel_3597626basemodel_3597628basemodel_3597630*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_35966422#
!basemodel/StatefulPartitionedCall
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/ConstФ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3597592*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs­
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1й
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_0_conv_1/kernel/Regularizer/mul/xм
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulй
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addЪ
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3597592*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square­
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2р
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1и
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/ConstФ
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3597588*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs­
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1й
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_1_conv_1/kernel/Regularizer/mul/xм
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulй
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addЪ
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3597588*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square­
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2р
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1и
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/ConstФ
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3597584*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs­
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1й
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_2_conv_1/kernel/Regularizer/mul/xм
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulй
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addЪ
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3597584*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square­
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2р
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1и
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/ConstБ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3597620*
_output_shapes
:	Р@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpЈ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2 
dense_1/kernel/Regularizer/Abs
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1Й
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_1/kernel/Regularizer/mul/xМ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addЗ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3597620*
_output_shapes
:	Р@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpД
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2#
!dense_1/kernel/Regularizer/Square
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2Р
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"dense_1/kernel/Regularizer/mul_1/xФ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1И
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

IdentityЎ
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:Y U
,
_output_shapes
:џџџџџџџџџњ
%
_user_specified_nameleft_inputs

s
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_3600017

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
:џџџџџџџџџ@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
И
Б
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3599654

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2

IdentityТ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
ўz
ј

D__inference_model_1_layer_call_and_return_conditional_losses_3597145

inputs'
basemodel_3597035:@
basemodel_3597037:@'
basemodel_3597039:@
basemodel_3597041:@'
basemodel_3597043:@
basemodel_3597045:@
basemodel_3597047:@
basemodel_3597049:@
basemodel_3597051:@
basemodel_3597053:@
basemodel_3597055:@
basemodel_3597057:@
basemodel_3597059:@
basemodel_3597061:@
basemodel_3597063:@
basemodel_3597065:@
basemodel_3597067:@
basemodel_3597069:@$
basemodel_3597071:	Р@
basemodel_3597073:@
basemodel_3597075:@
basemodel_3597077:@
basemodel_3597079:@
basemodel_3597081:@
identityЂ!basemodel/StatefulPartitionedCallЂ-dense_1/kernel/Regularizer/Abs/ReadVariableOpЂ0dense_1/kernel/Regularizer/Square/ReadVariableOpЂ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpџ
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsbasemodel_3597035basemodel_3597037basemodel_3597039basemodel_3597041basemodel_3597043basemodel_3597045basemodel_3597047basemodel_3597049basemodel_3597051basemodel_3597053basemodel_3597055basemodel_3597057basemodel_3597059basemodel_3597061basemodel_3597063basemodel_3597065basemodel_3597067basemodel_3597069basemodel_3597071basemodel_3597073basemodel_3597075basemodel_3597077basemodel_3597079basemodel_3597081*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_35959572#
!basemodel/StatefulPartitionedCall
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/ConstФ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3597043*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs­
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1й
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_0_conv_1/kernel/Regularizer/mul/xм
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulй
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addЪ
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3597043*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square­
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2р
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1и
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/ConstФ
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3597039*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs­
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1й
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_1_conv_1/kernel/Regularizer/mul/xм
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulй
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addЪ
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3597039*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square­
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2р
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1и
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/ConstФ
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3597035*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs­
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1й
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_2_conv_1/kernel/Regularizer/mul/xм
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulй
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addЪ
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3597035*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square­
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2р
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1и
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/ConstБ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3597071*
_output_shapes
:	Р@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpЈ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2 
dense_1/kernel/Regularizer/Abs
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1Й
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_1/kernel/Regularizer/mul/xМ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addЗ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3597071*
_output_shapes
:	Р@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpД
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2#
!dense_1/kernel/Regularizer/Square
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2Р
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"dense_1/kernel/Regularizer/mul_1/xФ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1И
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

IdentityЎ
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
Н
s
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_3600033

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
:џџџџџџџџџџџџџџџџџџ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
н
J
.__inference_activation_1_layer_call_fn_3599814

inputs
identityЯ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_35957552
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ@:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
щ
c
G__inference_activation_layer_call_and_return_conditional_losses_3599799

inputs
identityS
TanhTanhinputs*
T0*,
_output_shapes
:џџџџџџџџџњ@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ@:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
Й+
ы
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3595120

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradientБ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
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
з#<2
AssignMovingAvg/decayЄ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulП
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
з#<2
AssignMovingAvg_1/decayЊ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mulЩ
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2

Identityђ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs

n
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_3599076

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
Ю
n
5__inference_stream_0_input_drop_layer_call_fn_3599098

inputs
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_35963942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs

Б
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3599548

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

IdentityТ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџњ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs

j
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_3599961

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs

h
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_3596036

inputs
identity[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџР:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
Ќ
k
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_3595771

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@2

ExpandDims
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2	
MaxPool|
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ@:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
ю
k
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_3599973

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeг
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
dtype0*
seedЗ*
seed2К2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yТ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ}@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs

n
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_3599103

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
ы
Q
5__inference_stream_2_input_drop_layer_call_fn_3599147

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_35955322
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs

h
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_3600073

inputs
identity[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџР:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs

k
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_3599884

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

ExpandDimsБ
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ЉЫ
ђ
F__inference_basemodel_layer_call_and_return_conditional_losses_3598665
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
&dense_1_matmul_readvariableop_resource:	Р@5
'dense_1_biasadd_readvariableop_resource:@E
7batch_normalization_3_batchnorm_readvariableop_resource:@I
;batch_normalization_3_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_3_batchnorm_readvariableop_1_resource:@G
9batch_normalization_3_batchnorm_readvariableop_2_resource:@
identityЂ,batch_normalization/batchnorm/ReadVariableOpЂ.batch_normalization/batchnorm/ReadVariableOp_1Ђ.batch_normalization/batchnorm/ReadVariableOp_2Ђ0batch_normalization/batchnorm/mul/ReadVariableOpЂ.batch_normalization_1/batchnorm/ReadVariableOpЂ0batch_normalization_1/batchnorm/ReadVariableOp_1Ђ0batch_normalization_1/batchnorm/ReadVariableOp_2Ђ2batch_normalization_1/batchnorm/mul/ReadVariableOpЂ.batch_normalization_2/batchnorm/ReadVariableOpЂ0batch_normalization_2/batchnorm/ReadVariableOp_1Ђ0batch_normalization_2/batchnorm/ReadVariableOp_2Ђ2batch_normalization_2/batchnorm/mul/ReadVariableOpЂ.batch_normalization_3/batchnorm/ReadVariableOpЂ0batch_normalization_3/batchnorm/ReadVariableOp_1Ђ0batch_normalization_3/batchnorm/ReadVariableOp_2Ђ2batch_normalization_3/batchnorm/mul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂ-dense_1/kernel/Regularizer/Abs/ReadVariableOpЂ0dense_1/kernel/Regularizer/Square/ReadVariableOpЂ&stream_0_conv_1/BiasAdd/ReadVariableOpЂ2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpЂ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ&stream_1_conv_1/BiasAdd/ReadVariableOpЂ2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpЂ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ&stream_2_conv_1/BiasAdd/ReadVariableOpЂ2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpЂ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp
stream_2_input_drop/IdentityIdentityinputs_2*
T0*,
_output_shapes
:џџџџџџџџџњ2
stream_2_input_drop/Identity
stream_1_input_drop/IdentityIdentityinputs_1*
T0*,
_output_shapes
:џџџџџџџџџњ2
stream_1_input_drop/Identity
stream_0_input_drop/IdentityIdentityinputs_0*
T0*,
_output_shapes
:џџџџџџџџџњ2
stream_0_input_drop/Identity
%stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2'
%stream_2_conv_1/conv1d/ExpandDims/dimц
!stream_2_conv_1/conv1d/ExpandDims
ExpandDims%stream_2_input_drop/Identity:output:0.stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ2#
!stream_2_conv_1/conv1d/ExpandDimsш
2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp
'stream_2_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_2_conv_1/conv1d/ExpandDims_1/dimї
#stream_2_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_2_conv_1/conv1d/ExpandDims_1ї
stream_2_conv_1/conv1dConv2D*stream_2_conv_1/conv1d/ExpandDims:output:0,stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2
stream_2_conv_1/conv1dУ
stream_2_conv_1/conv1d/SqueezeSqueezestream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ2 
stream_2_conv_1/conv1d/SqueezeМ
&stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_2_conv_1/BiasAdd/ReadVariableOpЭ
stream_2_conv_1/BiasAddBiasAdd'stream_2_conv_1/conv1d/Squeeze:output:0.stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
stream_2_conv_1/BiasAdd
%stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2'
%stream_1_conv_1/conv1d/ExpandDims/dimц
!stream_1_conv_1/conv1d/ExpandDims
ExpandDims%stream_1_input_drop/Identity:output:0.stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ2#
!stream_1_conv_1/conv1d/ExpandDimsш
2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp
'stream_1_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_1_conv_1/conv1d/ExpandDims_1/dimї
#stream_1_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_1_conv_1/conv1d/ExpandDims_1ї
stream_1_conv_1/conv1dConv2D*stream_1_conv_1/conv1d/ExpandDims:output:0,stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2
stream_1_conv_1/conv1dУ
stream_1_conv_1/conv1d/SqueezeSqueezestream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ2 
stream_1_conv_1/conv1d/SqueezeМ
&stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_1_conv_1/BiasAdd/ReadVariableOpЭ
stream_1_conv_1/BiasAddBiasAdd'stream_1_conv_1/conv1d/Squeeze:output:0.stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
stream_1_conv_1/BiasAdd
%stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2'
%stream_0_conv_1/conv1d/ExpandDims/dimц
!stream_0_conv_1/conv1d/ExpandDims
ExpandDims%stream_0_input_drop/Identity:output:0.stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ2#
!stream_0_conv_1/conv1d/ExpandDimsш
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype024
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp
'stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_1/conv1d/ExpandDims_1/dimї
#stream_0_conv_1/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2%
#stream_0_conv_1/conv1d/ExpandDims_1ї
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2
stream_0_conv_1/conv1dУ
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ2 
stream_0_conv_1/conv1d/SqueezeМ
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOpЭ
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
stream_0_conv_1/BiasAddд
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_2/batchnorm/add/yр
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/addЅ
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpн
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/mulз
%batch_normalization_2/batchnorm/mul_1Mul stream_2_conv_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2'
%batch_normalization_2/batchnorm/mul_1к
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1н
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/mul_2к
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2л
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/subт
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2'
%batch_normalization_2/batchnorm/add_1д
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_1/batchnorm/add/yр
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/addЅ
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_1/batchnorm/Rsqrtр
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOpн
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/mulз
%batch_normalization_1/batchnorm/mul_1Mul stream_1_conv_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2'
%batch_normalization_1/batchnorm/mul_1к
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1н
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_1/batchnorm/mul_2к
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2л
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_1/batchnorm/subт
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2'
%batch_normalization_1/batchnorm/add_1Ю
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2%
#batch_normalization/batchnorm/add/yи
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/add
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/Rsqrtк
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpе
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/mulб
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2%
#batch_normalization/batchnorm/mul_1д
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1е
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2%
#batch_normalization/batchnorm/mul_2д
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2г
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2#
!batch_normalization/batchnorm/subк
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2%
#batch_normalization/batchnorm/add_1
activation_2/TanhTanh)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
activation_2/Tanh
activation_1/TanhTanh)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
activation_1/Tanh
activation/TanhTanh'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
activation/Tanh
!stream_2_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_2_maxpool_1/ExpandDims/dimЪ
stream_2_maxpool_1/ExpandDims
ExpandDimsactivation_2/Tanh:y:0*stream_2_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@2
stream_2_maxpool_1/ExpandDimsи
stream_2_maxpool_1/MaxPoolMaxPool&stream_2_maxpool_1/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2
stream_2_maxpool_1/MaxPoolЕ
stream_2_maxpool_1/SqueezeSqueeze#stream_2_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2
stream_2_maxpool_1/Squeeze
!stream_1_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_1_maxpool_1/ExpandDims/dimЪ
stream_1_maxpool_1/ExpandDims
ExpandDimsactivation_1/Tanh:y:0*stream_1_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@2
stream_1_maxpool_1/ExpandDimsи
stream_1_maxpool_1/MaxPoolMaxPool&stream_1_maxpool_1/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2
stream_1_maxpool_1/MaxPoolЕ
stream_1_maxpool_1/SqueezeSqueeze#stream_1_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2
stream_1_maxpool_1/Squeeze
!stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_1/ExpandDims/dimШ
stream_0_maxpool_1/ExpandDims
ExpandDimsactivation/Tanh:y:0*stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@2
stream_0_maxpool_1/ExpandDimsи
stream_0_maxpool_1/MaxPoolMaxPool&stream_0_maxpool_1/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_1/MaxPoolЕ
stream_0_maxpool_1/SqueezeSqueeze#stream_0_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2
stream_0_maxpool_1/Squeeze
stream_2_drop_1/IdentityIdentity#stream_2_maxpool_1/Squeeze:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
stream_2_drop_1/Identity
stream_1_drop_1/IdentityIdentity#stream_1_maxpool_1/Squeeze:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
stream_1_drop_1/Identity
stream_0_drop_1/IdentityIdentity#stream_0_maxpool_1/Squeeze:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
stream_0_drop_1/IdentityЄ
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indicesе
global_average_pooling1d/MeanMean!stream_0_drop_1/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
global_average_pooling1d/MeanЈ
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_1/Mean/reduction_indicesл
global_average_pooling1d_1/MeanMean!stream_1_drop_1/Identity:output:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2!
global_average_pooling1d_1/MeanЈ
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indicesл
global_average_pooling1d_2/MeanMean!stream_2_drop_1/Identity:output:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2!
global_average_pooling1d_2/Meant
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis
concatenate/concatConcatV2&global_average_pooling1d/Mean:output:0(global_average_pooling1d_1/Mean:output:0(global_average_pooling1d_2/Mean:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџР2
concatenate/concat
dense_1_dropout/IdentityIdentityconcatenate/concat:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
dense_1_dropout/IdentityІ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02
dense_1/MatMul/ReadVariableOpІ
dense_1/MatMulMatMul!dense_1_dropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_1/MatMulЄ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOpЁ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_1/BiasAddд
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_3/batchnorm/add/yр
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/addЅ
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/Rsqrtр
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpн
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/mulЪ
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2'
%batch_normalization_3/batchnorm/mul_1к
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1н
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/mul_2к
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2л
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/subн
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2'
%batch_normalization_3/batchnorm/add_1
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constю
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs­
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1й
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_0_conv_1/kernel/Regularizer/mul/xм
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulй
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addє
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square­
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2р
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1и
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Constю
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs­
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1й
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_1_conv_1/kernel/Regularizer/mul/xм
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulй
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addє
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square­
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2р
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1и
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Constю
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs­
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1й
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_2_conv_1/kernel/Regularizer/mul/xм
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulй
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addє
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square­
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2р
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1и
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/ConstЦ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpЈ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2 
dense_1/kernel/Regularizer/Abs
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1Й
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_1/kernel/Regularizer/mul/xМ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addЬ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpД
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2#
!dense_1/kernel/Regularizer/Square
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2Р
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"dense_1/kernel/Regularizer/mul_1/xФ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1И
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1
IdentityIdentity)batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_1_conv_1/BiasAdd/ReadVariableOp3^stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_2_conv_1/BiasAdd/ReadVariableOp3^stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesz
x:џџџџџџџџџњ:џџџџџџџџџњ:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 2\
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
&stream_1_conv_1/BiasAdd/ReadVariableOp&stream_1_conv_1/BiasAdd/ReadVariableOp2h
2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2P
&stream_2_conv_1/BiasAdd/ReadVariableOp&stream_2_conv_1/BiasAdd/ReadVariableOp2h
2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:V R
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs/2
ы
Q
5__inference_stream_1_input_drop_layer_call_fn_3599120

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_35955392
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_2_layer_call_fn_3599768

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_35951202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
іz
ј

D__inference_model_1_layer_call_and_return_conditional_losses_3597364

inputs'
basemodel_3597254:@
basemodel_3597256:@'
basemodel_3597258:@
basemodel_3597260:@'
basemodel_3597262:@
basemodel_3597264:@
basemodel_3597266:@
basemodel_3597268:@
basemodel_3597270:@
basemodel_3597272:@
basemodel_3597274:@
basemodel_3597276:@
basemodel_3597278:@
basemodel_3597280:@
basemodel_3597282:@
basemodel_3597284:@
basemodel_3597286:@
basemodel_3597288:@$
basemodel_3597290:	Р@
basemodel_3597292:@
basemodel_3597294:@
basemodel_3597296:@
basemodel_3597298:@
basemodel_3597300:@
identityЂ!basemodel/StatefulPartitionedCallЂ-dense_1/kernel/Regularizer/Abs/ReadVariableOpЂ0dense_1/kernel/Regularizer/Square/ReadVariableOpЂ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpї
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsbasemodel_3597254basemodel_3597256basemodel_3597258basemodel_3597260basemodel_3597262basemodel_3597264basemodel_3597266basemodel_3597268basemodel_3597270basemodel_3597272basemodel_3597274basemodel_3597276basemodel_3597278basemodel_3597280basemodel_3597282basemodel_3597284basemodel_3597286basemodel_3597288basemodel_3597290basemodel_3597292basemodel_3597294basemodel_3597296basemodel_3597298basemodel_3597300*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_35966422#
!basemodel/StatefulPartitionedCall
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/ConstФ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3597262*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs­
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1й
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_0_conv_1/kernel/Regularizer/mul/xм
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulй
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addЪ
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3597262*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square­
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2р
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1и
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/ConstФ
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3597258*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs­
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1й
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_1_conv_1/kernel/Regularizer/mul/xм
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulй
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addЪ
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3597258*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square­
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2р
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1и
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/ConstФ
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3597254*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs­
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1й
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_2_conv_1/kernel/Regularizer/mul/xм
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulй
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addЪ
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3597254*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square­
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2р
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1и
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/ConstБ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3597290*
_output_shapes
:	Р@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpЈ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2 
dense_1/kernel/Regularizer/Abs
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1Й
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_1/kernel/Regularizer/mul/xМ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addЗ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3597290*
_output_shapes
:	Р@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpД
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2#
!dense_1/kernel/Regularizer/Square
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2Р
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"dense_1/kernel/Regularizer/mul_1/xФ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1И
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

IdentityЎ
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs

k
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_3600216

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ќ
k
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_3599892

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@2

ExpandDims
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2	
MaxPool|
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ@:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
Ю
n
5__inference_stream_1_input_drop_layer_call_fn_3599125

inputs
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_35964172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
Э*
ы
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3600186

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
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
moments/StopGradientЄ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indicesВ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
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
з#<2
AssignMovingAvg/decayЄ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulП
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
з#<2
AssignMovingAvg_1/decayЊ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mulЩ
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityђ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
њ
o
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_3596417

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeд
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ*
dtype0*
seedЗ*
seed2И2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yУ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџњ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџњ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
ш

H__inference_concatenate_layer_call_and_return_conditional_losses_3600057
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџР2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ@:џџџџџџџџџ@:џџџџџџџџџ@:Q M
'
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
inputs/2
А
ч
)__inference_model_1_layer_call_fn_3597196
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

unknown_17:	Р@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityЂStatefulPartitionedCallЇ
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
:џџџџџџџџџ@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_35971452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:џџџџџџџџџњ
%
_user_specified_nameleft_inputs
Н
s
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_3600011

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
:џџџџџџџџџџџџџџџџџџ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ю
k
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_3596128

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeг
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
dtype0*
seedЗ*
seed2К2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yТ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ}@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
њ,

L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_3599197

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂ"conv1d/ExpandDims_1/ReadVariableOpЂ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ2
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
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2	
BiasAdd
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constо
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs­
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1й
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_0_conv_1/kernel/Regularizer/mul/xм
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulй
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addф
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square­
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2р
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1и
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1p
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityџ
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџњ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs

k
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_3599832

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

ExpandDimsБ
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ч
P
4__inference_stream_1_maxpool_1_layer_call_fn_3599876

inputs
identityд
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_35957802
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ@:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
	
а
5__inference_batch_normalization_layer_call_fn_3599435

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_35947362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
ј
љ
__inference_loss_fn_1_3600261T
>stream_1_conv_1_kernel_regularizer_abs_readvariableop_resource:@
identityЂ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Constё
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_1_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs­
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1й
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_1_conv_1/kernel/Regularizer/mul/xм
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulй
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addї
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>stream_1_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square­
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2р
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1и
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1v
IdentityIdentity,stream_1_conv_1/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityС
NoOpNoOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp
{
§

D__inference_model_1_layer_call_and_return_conditional_losses_3597581
left_inputs'
basemodel_3597471:@
basemodel_3597473:@'
basemodel_3597475:@
basemodel_3597477:@'
basemodel_3597479:@
basemodel_3597481:@
basemodel_3597483:@
basemodel_3597485:@
basemodel_3597487:@
basemodel_3597489:@
basemodel_3597491:@
basemodel_3597493:@
basemodel_3597495:@
basemodel_3597497:@
basemodel_3597499:@
basemodel_3597501:@
basemodel_3597503:@
basemodel_3597505:@$
basemodel_3597507:	Р@
basemodel_3597509:@
basemodel_3597511:@
basemodel_3597513:@
basemodel_3597515:@
basemodel_3597517:@
identityЂ!basemodel/StatefulPartitionedCallЂ-dense_1/kernel/Regularizer/Abs/ReadVariableOpЂ0dense_1/kernel/Regularizer/Square/ReadVariableOpЂ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsleft_inputsleft_inputsbasemodel_3597471basemodel_3597473basemodel_3597475basemodel_3597477basemodel_3597479basemodel_3597481basemodel_3597483basemodel_3597485basemodel_3597487basemodel_3597489basemodel_3597491basemodel_3597493basemodel_3597495basemodel_3597497basemodel_3597499basemodel_3597501basemodel_3597503basemodel_3597505basemodel_3597507basemodel_3597509basemodel_3597511basemodel_3597513basemodel_3597515basemodel_3597517*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_35959572#
!basemodel/StatefulPartitionedCall
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/ConstФ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3597479*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs­
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1й
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_0_conv_1/kernel/Regularizer/mul/xм
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulй
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addЪ
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3597479*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square­
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2р
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1и
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/ConstФ
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3597475*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs­
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1й
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_1_conv_1/kernel/Regularizer/mul/xм
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulй
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addЪ
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3597475*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square­
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2р
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1и
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/ConstФ
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3597471*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs­
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1й
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_2_conv_1/kernel/Regularizer/mul/xм
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulй
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addЪ
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3597471*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square­
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2р
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1и
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/ConstБ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_3597507*
_output_shapes
:	Р@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpЈ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2 
dense_1/kernel/Regularizer/Abs
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1Й
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_1/kernel/Regularizer/mul/xМ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addЗ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_3597507*
_output_shapes
:	Р@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpД
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2#
!dense_1/kernel/Regularizer/Square
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2Р
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"dense_1/kernel/Regularizer/mul_1/xФ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1И
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

IdentityЎ
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:Y U
,
_output_shapes
:џџџџџџџџџњ
%
_user_specified_nameleft_inputs
§
j
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_3600069

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџР2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџР:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs

j
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_3599934

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
њ,

L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_3595578

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂ"conv1d/ExpandDims_1/ReadVariableOpЂ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ2
conv1d/ExpandDimsИ
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
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2	
BiasAdd
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Constо
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs­
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1й
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_2_conv_1/kernel/Regularizer/mul/xм
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulй
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addф
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square­
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2р
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1и
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1p
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityџ
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџњ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
њ
o
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_3599088

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeд
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ*
dtype0*
seedЗ*
seed2З2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yУ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџњ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџњ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs

n
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_3599130

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs

n
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_3595532

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_1_layer_call_fn_3599595

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35948982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
§!
й
D__inference_dense_1_layer_call_and_return_conditional_losses_3595875

inputs1
matmul_readvariableop_resource:	Р@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ-dense_1/kernel/Regularizer/Abs/ReadVariableOpЂ0dense_1/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAdd
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/ConstО
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpЈ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2 
dense_1/kernel/Regularizer/Abs
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1Й
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_1/kernel/Regularizer/mul/xМ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addФ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpД
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2#
!dense_1/kernel/Regularizer/Square
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2Р
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"dense_1/kernel/Regularizer/mul_1/xФ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1И
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityт
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџР: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
+
ы
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3596273

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradientЉ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
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
з#<2
AssignMovingAvg/decayЄ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulП
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
з#<2
AssignMovingAvg_1/decayЊ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mulЩ
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityђ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџњ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs

n
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_3595546

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
Ў
P
4__inference_stream_2_maxpool_1_layer_call_fn_3599897

inputs
identityц
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_35952662
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_2_layer_call_fn_3599755

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_35950602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Ж

+__inference_basemodel_layer_call_fn_3599071
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

unknown_17:	Р@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityЂStatefulPartitionedCallД
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
:џџџџџџџџџ@*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_35966422
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesz
x:џџџџџџџџџњ:џџџџџџџџџњ:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs/2

s
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_3595824

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
:џџџџџџџџџ@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
Ен
Њ
F__inference_basemodel_layer_call_and_return_conditional_losses_3596888
inputs_0
inputs_1
inputs_2-
stream_2_conv_1_3596756:@%
stream_2_conv_1_3596758:@-
stream_1_conv_1_3596761:@%
stream_1_conv_1_3596763:@-
stream_0_conv_1_3596766:@%
stream_0_conv_1_3596768:@+
batch_normalization_2_3596771:@+
batch_normalization_2_3596773:@+
batch_normalization_2_3596775:@+
batch_normalization_2_3596777:@+
batch_normalization_1_3596780:@+
batch_normalization_1_3596782:@+
batch_normalization_1_3596784:@+
batch_normalization_1_3596786:@)
batch_normalization_3596789:@)
batch_normalization_3596791:@)
batch_normalization_3596793:@)
batch_normalization_3596795:@"
dense_1_3596812:	Р@
dense_1_3596814:@+
batch_normalization_3_3596817:@+
batch_normalization_3_3596819:@+
batch_normalization_3_3596821:@+
batch_normalization_3_3596823:@
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂ-dense_1/kernel/Regularizer/Abs/ReadVariableOpЂ0dense_1/kernel/Regularizer/Square/ReadVariableOpЂ'stream_0_conv_1/StatefulPartitionedCallЂ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ'stream_1_conv_1/StatefulPartitionedCallЂ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ'stream_2_conv_1/StatefulPartitionedCallЂ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp
#stream_2_input_drop/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_35955322%
#stream_2_input_drop/PartitionedCall
#stream_1_input_drop/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_35955392%
#stream_1_input_drop/PartitionedCall
#stream_0_input_drop/PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_35955462%
#stream_0_input_drop/PartitionedCallш
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_2_input_drop/PartitionedCall:output:0stream_2_conv_1_3596756stream_2_conv_1_3596758*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_35955782)
'stream_2_conv_1/StatefulPartitionedCallш
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_1_input_drop/PartitionedCall:output:0stream_1_conv_1_3596761stream_1_conv_1_3596763*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_35956142)
'stream_1_conv_1/StatefulPartitionedCallш
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_3596766stream_0_conv_1_3596768*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_35956502)
'stream_0_conv_1/StatefulPartitionedCallЬ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_3596771batch_normalization_2_3596773batch_normalization_2_3596775batch_normalization_2_3596777*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_35956752/
-batch_normalization_2/StatefulPartitionedCallЬ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_3596780batch_normalization_1_3596782batch_normalization_1_3596784batch_normalization_1_3596786*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35957042/
-batch_normalization_1/StatefulPartitionedCallО
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_3596789batch_normalization_3596791batch_normalization_3596793batch_normalization_3596795*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_35957332-
+batch_normalization/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_35957482
activation_2/PartitionedCall
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_35957552
activation_1/PartitionedCall
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_35957622
activation/PartitionedCall
"stream_2_maxpool_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_35957712$
"stream_2_maxpool_1/PartitionedCall
"stream_1_maxpool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_35957802$
"stream_1_maxpool_1/PartitionedCall
"stream_0_maxpool_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_35957892$
"stream_0_maxpool_1/PartitionedCall
stream_2_drop_1/PartitionedCallPartitionedCall+stream_2_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_35957962!
stream_2_drop_1/PartitionedCall
stream_1_drop_1/PartitionedCallPartitionedCall+stream_1_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_35958032!
stream_1_drop_1/PartitionedCall
stream_0_drop_1/PartitionedCallPartitionedCall+stream_0_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_35958102!
stream_0_drop_1/PartitionedCallЊ
(global_average_pooling1d/PartitionedCallPartitionedCall(stream_0_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_35958172*
(global_average_pooling1d/PartitionedCallА
*global_average_pooling1d_1/PartitionedCallPartitionedCall(stream_1_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_35958242,
*global_average_pooling1d_1/PartitionedCallА
*global_average_pooling1d_2/PartitionedCallPartitionedCall(stream_2_drop_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_35958312,
*global_average_pooling1d_2/PartitionedCallљ
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:03global_average_pooling1d_1/PartitionedCall:output:03global_average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_35958412
concatenate/PartitionedCall
dense_1_dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_35958482!
dense_1_dropout/PartitionedCallЗ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_3596812dense_1_3596814*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_35958752!
dense_1/StatefulPartitionedCallП
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_3596817batch_normalization_3_3596819batch_normalization_3_3596821batch_normalization_3_3596823*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_35953782/
-batch_normalization_3/StatefulPartitionedCallІ
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_35958942$
"dense_activation_1/PartitionedCall
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/ConstЪ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_3596766*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs­
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1й
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_0_conv_1/kernel/Regularizer/mul/xм
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulй
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addа
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_1_3596766*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square­
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2р
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1и
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/ConstЪ
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_1_conv_1_3596761*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs­
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1й
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_1_conv_1/kernel/Regularizer/mul/xм
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulй
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addа
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_3596761*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square­
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2р
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1и
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/ConstЪ
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_3596756*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs­
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1й
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_2_conv_1/kernel/Regularizer/mul/xм
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulй
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addа
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_2_conv_1_3596756*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square­
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2р
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1и
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/ConstЏ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_3596812*
_output_shapes
:	Р@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpЈ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2 
dense_1/kernel/Regularizer/Abs
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1Й
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_1/kernel/Regularizer/mul/xМ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addЕ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_3596812*
_output_shapes
:	Р@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpД
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2#
!dense_1/kernel/Regularizer/Square
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2Р
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"dense_1/kernel/Regularizer/mul_1/xФ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1И
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityш
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_1_conv_1/StatefulPartitionedCall6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesz
x:џџџџџџџџџњ:џџџџџџџџџњ:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
'stream_1_conv_1/StatefulPartitionedCall'stream_1_conv_1/StatefulPartitionedCall2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2R
'stream_2_conv_1/StatefulPartitionedCall'stream_2_conv_1/StatefulPartitionedCall2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:V R
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs_2
г
M
1__inference_dense_1_dropout_layer_call_fn_3600083

inputs
identityЮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_35960362
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџР:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
Й+
ы
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3599528

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradientБ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
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
з#<2
AssignMovingAvg/decayЄ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulП
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
з#<2
AssignMovingAvg_1/decayЊ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mulЩ
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2

Identityђ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
џю
В
D__inference_model_1_layer_call_and_return_conditional_losses_3598006

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
0basemodel_dense_1_matmul_readvariableop_resource:	Р@?
1basemodel_dense_1_biasadd_readvariableop_resource:@O
Abasemodel_batch_normalization_3_batchnorm_readvariableop_resource:@S
Ebasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:@Q
Cbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource:@Q
Cbasemodel_batch_normalization_3_batchnorm_readvariableop_2_resource:@
identityЂ6basemodel/batch_normalization/batchnorm/ReadVariableOpЂ8basemodel/batch_normalization/batchnorm/ReadVariableOp_1Ђ8basemodel/batch_normalization/batchnorm/ReadVariableOp_2Ђ:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpЂ8basemodel/batch_normalization_1/batchnorm/ReadVariableOpЂ:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1Ђ:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2Ђ<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpЂ8basemodel/batch_normalization_2/batchnorm/ReadVariableOpЂ:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1Ђ:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2Ђ<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpЂ8basemodel/batch_normalization_3/batchnorm/ReadVariableOpЂ:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1Ђ:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2Ђ<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpЂ(basemodel/dense_1/BiasAdd/ReadVariableOpЂ'basemodel/dense_1/MatMul/ReadVariableOpЂ0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpЂ<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpЂ0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpЂ<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpЂ0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpЂ<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpЂ-dense_1/kernel/Regularizer/Abs/ReadVariableOpЂ0dense_1/kernel/Regularizer/Square/ReadVariableOpЂ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЂ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpЂ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp
&basemodel/stream_2_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:џџџџџџџџџњ2(
&basemodel/stream_2_input_drop/Identity
&basemodel/stream_1_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:џџџџџџџџџњ2(
&basemodel/stream_1_input_drop/Identity
&basemodel/stream_0_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:џџџџџџџџџњ2(
&basemodel/stream_0_input_drop/Identity­
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ21
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dim
+basemodel/stream_2_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_2_input_drop/Identity:output:08basemodel/stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ2-
+basemodel/stream_2_conv_1/conv1d/ExpandDims
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpЈ
1basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dim
-basemodel/stream_2_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_2_conv_1/conv1d/ExpandDims_1
 basemodel/stream_2_conv_1/conv1dConv2D4basemodel/stream_2_conv_1/conv1d/ExpandDims:output:06basemodel/stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2"
 basemodel/stream_2_conv_1/conv1dс
(basemodel/stream_2_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ2*
(basemodel/stream_2_conv_1/conv1d/Squeezeк
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpѕ
!basemodel/stream_2_conv_1/BiasAddBiasAdd1basemodel/stream_2_conv_1/conv1d/Squeeze:output:08basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2#
!basemodel/stream_2_conv_1/BiasAdd­
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ21
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dim
+basemodel/stream_1_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_1_input_drop/Identity:output:08basemodel/stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ2-
+basemodel/stream_1_conv_1/conv1d/ExpandDims
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpЈ
1basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dim
-basemodel/stream_1_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_1_conv_1/conv1d/ExpandDims_1
 basemodel/stream_1_conv_1/conv1dConv2D4basemodel/stream_1_conv_1/conv1d/ExpandDims:output:06basemodel/stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2"
 basemodel/stream_1_conv_1/conv1dс
(basemodel/stream_1_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ2*
(basemodel/stream_1_conv_1/conv1d/Squeezeк
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpѕ
!basemodel/stream_1_conv_1/BiasAddBiasAdd1basemodel/stream_1_conv_1/conv1d/Squeeze:output:08basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2#
!basemodel/stream_1_conv_1/BiasAdd­
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ21
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dim
+basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_0_input_drop/Identity:output:08basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ2-
+basemodel/stream_0_conv_1/conv1d/ExpandDims
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02>
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpЈ
1basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1
ExpandDimsDbasemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2/
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1
 basemodel/stream_0_conv_1/conv1dConv2D4basemodel/stream_0_conv_1/conv1d/ExpandDims:output:06basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1dс
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ2*
(basemodel/stream_0_conv_1/conv1d/Squeezeк
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpѕ
!basemodel/stream_0_conv_1/BiasAddBiasAdd1basemodel/stream_0_conv_1/conv1d/Squeeze:output:08basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2#
!basemodel/stream_0_conv_1/BiasAddђ
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpЇ
/basemodel/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:21
/basemodel/batch_normalization_2/batchnorm/add/y
-basemodel/batch_normalization_2/batchnorm/addAddV2@basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:08basemodel/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/addУ
/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm/Rsqrtў
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp
-basemodel/batch_normalization_2/batchnorm/mulMul3basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/mulџ
/basemodel/batch_normalization_2/batchnorm/mul_1Mul*basemodel/stream_2_conv_1/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@21
/basemodel/batch_normalization_2/batchnorm/mul_1ј
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1
/basemodel/batch_normalization_2/batchnorm/mul_2MulBbasemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_2/batchnorm/mul_2ј
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2
-basemodel/batch_normalization_2/batchnorm/subSubBbasemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_2/batchnorm/sub
/basemodel/batch_normalization_2/batchnorm/add_1AddV23basemodel/batch_normalization_2/batchnorm/mul_1:z:01basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@21
/basemodel/batch_normalization_2/batchnorm/add_1ђ
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_1/batchnorm/ReadVariableOpЇ
/basemodel/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:21
/basemodel/batch_normalization_1/batchnorm/add/y
-basemodel/batch_normalization_1/batchnorm/addAddV2@basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:08basemodel/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/addУ
/basemodel/batch_normalization_1/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm/Rsqrtў
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp
-basemodel/batch_normalization_1/batchnorm/mulMul3basemodel/batch_normalization_1/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/mulџ
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_1_conv_1/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@21
/basemodel/batch_normalization_1/batchnorm/mul_1ј
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1
/basemodel/batch_normalization_1/batchnorm/mul_2MulBbasemodel/batch_normalization_1/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm/mul_2ј
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2
-basemodel/batch_normalization_1/batchnorm/subSubBbasemodel/batch_normalization_1/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_1/batchnorm/sub
/basemodel/batch_normalization_1/batchnorm/add_1AddV23basemodel/batch_normalization_1/batchnorm/mul_1:z:01basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@21
/basemodel/batch_normalization_1/batchnorm/add_1ь
6basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype028
6basemodel/batch_normalization/batchnorm/ReadVariableOpЃ
-basemodel/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-basemodel/batch_normalization/batchnorm/add/y
+basemodel/batch_normalization/batchnorm/addAddV2>basemodel/batch_normalization/batchnorm/ReadVariableOp:value:06basemodel/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/addН
-basemodel/batch_normalization/batchnorm/RsqrtRsqrt/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/Rsqrtј
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp§
+basemodel/batch_normalization/batchnorm/mulMul1basemodel/batch_normalization/batchnorm/Rsqrt:y:0Bbasemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/mulљ
-basemodel/batch_normalization/batchnorm/mul_1Mul*basemodel/stream_0_conv_1/BiasAdd:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2/
-basemodel/batch_normalization/batchnorm/mul_1ђ
8basemodel/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization/batchnorm/ReadVariableOp_1§
-basemodel/batch_normalization/batchnorm/mul_2Mul@basemodel/batch_normalization/batchnorm/ReadVariableOp_1:value:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization/batchnorm/mul_2ђ
8basemodel/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization/batchnorm/ReadVariableOp_2ћ
+basemodel/batch_normalization/batchnorm/subSub@basemodel/batch_normalization/batchnorm/ReadVariableOp_2:value:01basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2-
+basemodel/batch_normalization/batchnorm/sub
-basemodel/batch_normalization/batchnorm/add_1AddV21basemodel/batch_normalization/batchnorm/mul_1:z:0/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2/
-basemodel/batch_normalization/batchnorm/add_1Ў
basemodel/activation_2/TanhTanh3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
basemodel/activation_2/TanhЎ
basemodel/activation_1/TanhTanh3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
basemodel/activation_1/TanhЈ
basemodel/activation/TanhTanh1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
basemodel/activation/Tanh
+basemodel/stream_2_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_2_maxpool_1/ExpandDims/dimђ
'basemodel/stream_2_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation_2/Tanh:y:04basemodel/stream_2_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@2)
'basemodel/stream_2_maxpool_1/ExpandDimsі
$basemodel/stream_2_maxpool_1/MaxPoolMaxPool0basemodel/stream_2_maxpool_1/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_2_maxpool_1/MaxPoolг
$basemodel/stream_2_maxpool_1/SqueezeSqueeze-basemodel/stream_2_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2&
$basemodel/stream_2_maxpool_1/Squeeze
+basemodel/stream_1_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_1_maxpool_1/ExpandDims/dimђ
'basemodel/stream_1_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation_1/Tanh:y:04basemodel/stream_1_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@2)
'basemodel/stream_1_maxpool_1/ExpandDimsі
$basemodel/stream_1_maxpool_1/MaxPoolMaxPool0basemodel/stream_1_maxpool_1/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_1_maxpool_1/MaxPoolг
$basemodel/stream_1_maxpool_1/SqueezeSqueeze-basemodel/stream_1_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2&
$basemodel/stream_1_maxpool_1/Squeeze
+basemodel/stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_0_maxpool_1/ExpandDims/dim№
'basemodel/stream_0_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation/Tanh:y:04basemodel/stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@2)
'basemodel/stream_0_maxpool_1/ExpandDimsі
$basemodel/stream_0_maxpool_1/MaxPoolMaxPool0basemodel/stream_0_maxpool_1/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ}@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_0_maxpool_1/MaxPoolг
$basemodel/stream_0_maxpool_1/SqueezeSqueeze-basemodel/stream_0_maxpool_1/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
squeeze_dims
2&
$basemodel/stream_0_maxpool_1/SqueezeЙ
"basemodel/stream_2_drop_1/IdentityIdentity-basemodel/stream_2_maxpool_1/Squeeze:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2$
"basemodel/stream_2_drop_1/IdentityЙ
"basemodel/stream_1_drop_1/IdentityIdentity-basemodel/stream_1_maxpool_1/Squeeze:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2$
"basemodel/stream_1_drop_1/IdentityЙ
"basemodel/stream_0_drop_1/IdentityIdentity-basemodel/stream_0_maxpool_1/Squeeze:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2$
"basemodel/stream_0_drop_1/IdentityИ
9basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9basemodel/global_average_pooling1d/Mean/reduction_indices§
'basemodel/global_average_pooling1d/MeanMean+basemodel/stream_0_drop_1/Identity:output:0Bbasemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2)
'basemodel/global_average_pooling1d/MeanМ
;basemodel/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d_1/Mean/reduction_indices
)basemodel/global_average_pooling1d_1/MeanMean+basemodel/stream_1_drop_1/Identity:output:0Dbasemodel/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2+
)basemodel/global_average_pooling1d_1/MeanМ
;basemodel/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d_2/Mean/reduction_indices
)basemodel/global_average_pooling1d_2/MeanMean+basemodel/stream_2_drop_1/Identity:output:0Dbasemodel/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2+
)basemodel/global_average_pooling1d_2/Mean
!basemodel/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!basemodel/concatenate/concat/axisЪ
basemodel/concatenate/concatConcatV20basemodel/global_average_pooling1d/Mean:output:02basemodel/global_average_pooling1d_1/Mean:output:02basemodel/global_average_pooling1d_2/Mean:output:0*basemodel/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџР2
basemodel/concatenate/concatЎ
"basemodel/dense_1_dropout/IdentityIdentity%basemodel/concatenate/concat:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2$
"basemodel/dense_1_dropout/IdentityФ
'basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02)
'basemodel/dense_1/MatMul/ReadVariableOpЮ
basemodel/dense_1/MatMulMatMul+basemodel/dense_1_dropout/Identity:output:0/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
basemodel/dense_1/MatMulТ
(basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(basemodel/dense_1/BiasAdd/ReadVariableOpЩ
basemodel/dense_1/BiasAddBiasAdd"basemodel/dense_1/MatMul:product:00basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
basemodel/dense_1/BiasAddђ
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02:
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpЇ
/basemodel/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:21
/basemodel/batch_normalization_3/batchnorm/add/y
-basemodel/batch_normalization_3/batchnorm/addAddV2@basemodel/batch_normalization_3/batchnorm/ReadVariableOp:value:08basemodel/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_3/batchnorm/addУ
/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_3/batchnorm/Rsqrtў
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp
-basemodel/batch_normalization_3/batchnorm/mulMul3basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_3/batchnorm/mulђ
/basemodel/batch_normalization_3/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@21
/basemodel/batch_normalization_3/batchnorm/mul_1ј
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1
/basemodel/batch_normalization_3/batchnorm/mul_2MulBbasemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_3/batchnorm/mul_2ј
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2
-basemodel/batch_normalization_3/batchnorm/subSubBbasemodel/batch_normalization_3/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_3/batchnorm/sub
/basemodel/batch_normalization_3/batchnorm/add_1AddV23basemodel/batch_normalization_3/batchnorm/mul_1:z:01basemodel/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@21
/basemodel/batch_normalization_3/batchnorm/add_1
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constј
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Abs­
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1й
&stream_0_conv_1/kernel/Regularizer/SumSum*stream_0_conv_1/kernel/Regularizer/Abs:y:03stream_0_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/Sum
(stream_0_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_0_conv_1/kernel/Regularizer/mul/xм
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mulй
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addў
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Square­
*stream_0_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_2р
(stream_0_conv_1/kernel/Regularizer/Sum_1Sum-stream_0_conv_1/kernel/Regularizer/Square:y:03stream_0_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/Sum_1
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1и
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Constј
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs­
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1й
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/Sum
(stream_1_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_1_conv_1/kernel/Regularizer/mul/xм
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mulй
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addў
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square­
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2р
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1и
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Constј
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpУ
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs­
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1й
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/Sum
(stream_2_conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2*
(stream_2_conv_1/kernel/Regularizer/mul/xм
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mulй
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addў
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square­
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2р
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1и
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Constа
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpЈ
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2 
dense_1/kernel/Regularizer/Abs
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1Й
dense_1/kernel/Regularizer/SumSum"dense_1/kernel/Regularizer/Abs:y:0+dense_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2"
 dense_1/kernel/Regularizer/mul/xМ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mulЙ
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/addж
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Р@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpД
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Р@2#
!dense_1/kernel/Regularizer/Square
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2Р
 dense_1/kernel/Regularizer/Sum_1Sum%dense_1/kernel/Regularizer/Square:y:0+dense_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/Sum_1
"dense_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"dense_1/kernel/Regularizer/mul_1/xФ
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1И
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1
IdentityIdentity3basemodel/batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity§
NoOpNoOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp9^basemodel/batch_normalization/batchnorm/ReadVariableOp_19^basemodel/batch_normalization/batchnorm/ReadVariableOp_2;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_3/batchnorm/ReadVariableOp;^basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 2p
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_1_layer_call_fn_3599608

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35949582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs

j
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_3595810

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
Й+
ы
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3599688

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradientБ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
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
з#<2
AssignMovingAvg/decayЄ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulП
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
з#<2
AssignMovingAvg_1/decayЊ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mulЩ
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2

Identityђ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs

Џ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3595733

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

IdentityТ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџњ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
Ў
P
4__inference_stream_1_maxpool_1_layer_call_fn_3599871

inputs
identityц
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_35952382
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ж

+__inference_basemodel_layer_call_fn_3596748
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

unknown_17:	Р@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityЂStatefulPartitionedCallД
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
:џџџџџџџџџ@*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_35966422
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesz
x:џџџџџџџџџњ:џџџџџџџџџњ:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:џџџџџџџџџњ
"
_user_specified_name
inputs_2
п
M
1__inference_stream_0_drop_1_layer_call_fn_3599924

inputs
identityб
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ}@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_35958102
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
Л
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3599989

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
:џџџџџџџџџџџџџџџџџџ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
+
щ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3599422

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradientЉ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
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
з#<2
AssignMovingAvg/decayЄ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulП
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
з#<2
AssignMovingAvg_1/decayЊ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mulЩ
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityђ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџњ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
ю
k
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_3596082

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЋЊЊ?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeг
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@*
dtype0*
seedЗ*
seed2И2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yТ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ}@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ}@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs

у
%__inference_signature_wrapper_3597809
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

unknown_17:	Р@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityЂStatefulPartitionedCall
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
:џџџџџџџџџ@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_35947122
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџњ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:џџџџџџџџџњ
%
_user_specified_nameleft_inputs
+
ы
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3596333

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradientЉ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЖ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze
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
з#<2
AssignMovingAvg/decayЄ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mulП
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
з#<2
AssignMovingAvg_1/decayЊ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mulЩ
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identityђ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџњ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
э
X
<__inference_global_average_pooling1d_1_layer_call_fn_3600027

inputs
identityи
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_35958242
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ}@:S O
+
_output_shapes
:џџџџџџџџџ}@
 
_user_specified_nameinputs
Ю
n
5__inference_stream_2_input_drop_layer_call_fn_3599152

inputs
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_35964402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
о

H__inference_concatenate_layer_call_and_return_conditional_losses_3595841

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџР2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ@:џџџџџџџџџ@:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ж
Џ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3599334

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@2

IdentityТ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџџџџџџџџџџ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
н
J
.__inference_activation_2_layer_call_fn_3599824

inputs
identityЯ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_35957482
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџњ@:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
і
Б
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3600152

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOp
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
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

IdentityТ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs"ЈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Й
serving_defaultЅ
H
left_inputs9
serving_default_left_inputs:0џџџџџџџџџњ=
	basemodel0
StatefulPartitionedCall:0џџџџџџџџџ@tensorflow/serving/predict:Иў

layer-0
layer_with_weights-0
layer-1
regularization_losses
trainable_variables
	variables
	keras_api

signatures
Й_default_save_signature
+К&call_and_return_all_conditional_losses
Л__call__"
_tf_keras_network
"
_tf_keras_input_layer

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
layer-23
 layer-24
!layer-25
"layer_with_weights-6
"layer-26
#layer_with_weights-7
#layer-27
$layer-28
%regularization_losses
&trainable_variables
'	variables
(	keras_api
+М&call_and_return_all_conditional_losses
Н__call__"
_tf_keras_network
 "
trackable_list_wrapper

)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815"
trackable_list_wrapper
ж
)0
*1
+2
,3
-4
.5
/6
07
98
:9
110
211
;12
<13
314
415
=16
>17
518
619
720
821
?22
@23"
trackable_list_wrapper
Ю
Anon_trainable_variables
Bmetrics
Clayer_regularization_losses
regularization_losses
Dlayer_metrics

Elayers
trainable_variables
	variables
Л__call__
Й_default_save_signature
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
-
Оserving_default"
signature_map
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Ї
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
+П&call_and_return_all_conditional_losses
Р__call__"
_tf_keras_layer
Ї
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
+С&call_and_return_all_conditional_losses
Т__call__"
_tf_keras_layer
Ї
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"
_tf_keras_layer
Н

)kernel
*bias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"
_tf_keras_layer
Н

+kernel
,bias
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
+Ч&call_and_return_all_conditional_losses
Ш__call__"
_tf_keras_layer
Н

-kernel
.bias
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
+Щ&call_and_return_all_conditional_losses
Ъ__call__"
_tf_keras_layer
ь
^axis
	/gamma
0beta
9moving_mean
:moving_variance
_regularization_losses
`trainable_variables
a	variables
b	keras_api
+Ы&call_and_return_all_conditional_losses
Ь__call__"
_tf_keras_layer
ь
caxis
	1gamma
2beta
;moving_mean
<moving_variance
dregularization_losses
etrainable_variables
f	variables
g	keras_api
+Э&call_and_return_all_conditional_losses
Ю__call__"
_tf_keras_layer
ь
haxis
	3gamma
4beta
=moving_mean
>moving_variance
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
+Я&call_and_return_all_conditional_losses
а__call__"
_tf_keras_layer
Ї
mregularization_losses
ntrainable_variables
o	variables
p	keras_api
+б&call_and_return_all_conditional_losses
в__call__"
_tf_keras_layer
Ї
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
+г&call_and_return_all_conditional_losses
д__call__"
_tf_keras_layer
Ї
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"
_tf_keras_layer
Ї
yregularization_losses
ztrainable_variables
{	variables
|	keras_api
+з&call_and_return_all_conditional_losses
и__call__"
_tf_keras_layer
Ј
}regularization_losses
~trainable_variables
	variables
	keras_api
+й&call_and_return_all_conditional_losses
к__call__"
_tf_keras_layer
Ћ
regularization_losses
trainable_variables
	variables
	keras_api
+л&call_and_return_all_conditional_losses
м__call__"
_tf_keras_layer
Ћ
regularization_losses
trainable_variables
	variables
	keras_api
+н&call_and_return_all_conditional_losses
о__call__"
_tf_keras_layer
Ћ
regularization_losses
trainable_variables
	variables
	keras_api
+п&call_and_return_all_conditional_losses
р__call__"
_tf_keras_layer
Ћ
regularization_losses
trainable_variables
	variables
	keras_api
+с&call_and_return_all_conditional_losses
т__call__"
_tf_keras_layer
Ћ
regularization_losses
trainable_variables
	variables
	keras_api
+у&call_and_return_all_conditional_losses
ф__call__"
_tf_keras_layer
Ћ
regularization_losses
trainable_variables
	variables
	keras_api
+х&call_and_return_all_conditional_losses
ц__call__"
_tf_keras_layer
Ћ
regularization_losses
trainable_variables
	variables
	keras_api
+ч&call_and_return_all_conditional_losses
ш__call__"
_tf_keras_layer
Ћ
regularization_losses
trainable_variables
	variables
 	keras_api
+щ&call_and_return_all_conditional_losses
ъ__call__"
_tf_keras_layer
Ћ
Ёregularization_losses
Ђtrainable_variables
Ѓ	variables
Є	keras_api
+ы&call_and_return_all_conditional_losses
ь__call__"
_tf_keras_layer
С

5kernel
6bias
Ѕregularization_losses
Іtrainable_variables
Ї	variables
Ј	keras_api
+э&call_and_return_all_conditional_losses
ю__call__"
_tf_keras_layer
ё
	Љaxis
	7gamma
8beta
?moving_mean
@moving_variance
Њregularization_losses
Ћtrainable_variables
Ќ	variables
­	keras_api
+я&call_and_return_all_conditional_losses
№__call__"
_tf_keras_layer
Ћ
Ўregularization_losses
Џtrainable_variables
А	variables
Б	keras_api
+ё&call_and_return_all_conditional_losses
ђ__call__"
_tf_keras_layer
@
ѓ0
є1
ѕ2
і3"
trackable_list_wrapper

)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815"
trackable_list_wrapper
ж
)0
*1
+2
,3
-4
.5
/6
07
98
:9
110
211
;12
<13
314
415
=16
>17
518
619
720
821
?22
@23"
trackable_list_wrapper
Е
Вnon_trainable_variables
Гmetrics
 Дlayer_regularization_losses
%regularization_losses
Еlayer_metrics
Жlayers
&trainable_variables
'	variables
Н__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
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
!:	Р@2dense_1/kernel
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
X
90
:1
;2
<3
=4
>5
?6
@7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
Е
Зmetrics
Иnon_trainable_variables
 Йlayer_regularization_losses
Fregularization_losses
Кlayer_metrics
Лlayers
Gtrainable_variables
H	variables
Р__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Мmetrics
Нnon_trainable_variables
 Оlayer_regularization_losses
Jregularization_losses
Пlayer_metrics
Рlayers
Ktrainable_variables
L	variables
Т__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Сmetrics
Тnon_trainable_variables
 Уlayer_regularization_losses
Nregularization_losses
Фlayer_metrics
Хlayers
Otrainable_variables
P	variables
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
(
ѓ0"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
Е
Цmetrics
Чnon_trainable_variables
 Шlayer_regularization_losses
Rregularization_losses
Щlayer_metrics
Ъlayers
Strainable_variables
T	variables
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
(
є0"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
Е
Ыmetrics
Ьnon_trainable_variables
 Эlayer_regularization_losses
Vregularization_losses
Юlayer_metrics
Яlayers
Wtrainable_variables
X	variables
Ш__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
(
ѕ0"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
Е
аmetrics
бnon_trainable_variables
 вlayer_regularization_losses
Zregularization_losses
гlayer_metrics
дlayers
[trainable_variables
\	variables
Ъ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
<
/0
01
92
:3"
trackable_list_wrapper
Е
еmetrics
жnon_trainable_variables
 зlayer_regularization_losses
_regularization_losses
иlayer_metrics
йlayers
`trainable_variables
a	variables
Ь__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
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
;2
<3"
trackable_list_wrapper
Е
кmetrics
лnon_trainable_variables
 мlayer_regularization_losses
dregularization_losses
нlayer_metrics
оlayers
etrainable_variables
f	variables
Ю__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
<
30
41
=2
>3"
trackable_list_wrapper
Е
пmetrics
рnon_trainable_variables
 сlayer_regularization_losses
iregularization_losses
тlayer_metrics
уlayers
jtrainable_variables
k	variables
а__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
фmetrics
хnon_trainable_variables
 цlayer_regularization_losses
mregularization_losses
чlayer_metrics
шlayers
ntrainable_variables
o	variables
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
щmetrics
ъnon_trainable_variables
 ыlayer_regularization_losses
qregularization_losses
ьlayer_metrics
эlayers
rtrainable_variables
s	variables
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
юmetrics
яnon_trainable_variables
 №layer_regularization_losses
uregularization_losses
ёlayer_metrics
ђlayers
vtrainable_variables
w	variables
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
ѓmetrics
єnon_trainable_variables
 ѕlayer_regularization_losses
yregularization_losses
іlayer_metrics
їlayers
ztrainable_variables
{	variables
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
јmetrics
љnon_trainable_variables
 њlayer_regularization_losses
}regularization_losses
ћlayer_metrics
ќlayers
~trainable_variables
	variables
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
И
§metrics
ўnon_trainable_variables
 џlayer_regularization_losses
regularization_losses
layer_metrics
layers
trainable_variables
	variables
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
И
metrics
non_trainable_variables
 layer_regularization_losses
regularization_losses
layer_metrics
layers
trainable_variables
	variables
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
И
metrics
non_trainable_variables
 layer_regularization_losses
regularization_losses
layer_metrics
layers
trainable_variables
	variables
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
И
metrics
non_trainable_variables
 layer_regularization_losses
regularization_losses
layer_metrics
layers
trainable_variables
	variables
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
И
metrics
non_trainable_variables
 layer_regularization_losses
regularization_losses
layer_metrics
layers
trainable_variables
	variables
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
И
metrics
non_trainable_variables
 layer_regularization_losses
regularization_losses
layer_metrics
layers
trainable_variables
	variables
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
И
metrics
non_trainable_variables
 layer_regularization_losses
regularization_losses
layer_metrics
layers
trainable_variables
	variables
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
И
 metrics
Ёnon_trainable_variables
 Ђlayer_regularization_losses
regularization_losses
Ѓlayer_metrics
Єlayers
trainable_variables
	variables
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
И
Ѕmetrics
Іnon_trainable_variables
 Їlayer_regularization_losses
Ёregularization_losses
Јlayer_metrics
Љlayers
Ђtrainable_variables
Ѓ	variables
ь__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
(
і0"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
И
Њmetrics
Ћnon_trainable_variables
 Ќlayer_regularization_losses
Ѕregularization_losses
­layer_metrics
Ўlayers
Іtrainable_variables
Ї	variables
ю__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
<
70
81
?2
@3"
trackable_list_wrapper
И
Џmetrics
Аnon_trainable_variables
 Бlayer_regularization_losses
Њregularization_losses
Вlayer_metrics
Гlayers
Ћtrainable_variables
Ќ	variables
№__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Дmetrics
Еnon_trainable_variables
 Жlayer_regularization_losses
Ўregularization_losses
Зlayer_metrics
Иlayers
Џtrainable_variables
А	variables
ђ__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
X
90
:1
;2
<3
=4
>5
?6
@7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ў
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
"26
#27
$28"
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
ѓ0"
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
є0"
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
ѕ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
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
;0
<1"
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
=0
>1"
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
і0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
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
бBЮ
"__inference__wrapped_model_3594712left_inputs"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
о2л
D__inference_model_1_layer_call_and_return_conditional_losses_3598006
D__inference_model_1_layer_call_and_return_conditional_losses_3598300
D__inference_model_1_layer_call_and_return_conditional_losses_3597581
D__inference_model_1_layer_call_and_return_conditional_losses_3597694Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ђ2я
)__inference_model_1_layer_call_fn_3597196
)__inference_model_1_layer_call_fn_3598353
)__inference_model_1_layer_call_fn_3598406
)__inference_model_1_layer_call_fn_3597468Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
F__inference_basemodel_layer_call_and_return_conditional_losses_3598665
F__inference_basemodel_layer_call_and_return_conditional_losses_3598961
F__inference_basemodel_layer_call_and_return_conditional_losses_3596888
F__inference_basemodel_layer_call_and_return_conditional_losses_3597028Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
њ2ї
+__inference_basemodel_layer_call_fn_3596008
+__inference_basemodel_layer_call_fn_3599016
+__inference_basemodel_layer_call_fn_3599071
+__inference_basemodel_layer_call_fn_3596748Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
аBЭ
%__inference_signature_wrapper_3597809left_inputs"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
о2л
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_3599076
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_3599088Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2Ѕ
5__inference_stream_0_input_drop_layer_call_fn_3599093
5__inference_stream_0_input_drop_layer_call_fn_3599098Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
о2л
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_3599103
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_3599115Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2Ѕ
5__inference_stream_1_input_drop_layer_call_fn_3599120
5__inference_stream_1_input_drop_layer_call_fn_3599125Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
о2л
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_3599130
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_3599142Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2Ѕ
5__inference_stream_2_input_drop_layer_call_fn_3599147
5__inference_stream_2_input_drop_layer_call_fn_3599152Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
і2ѓ
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_3599197Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
л2и
1__inference_stream_0_conv_1_layer_call_fn_3599206Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_3599251Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
л2и
1__inference_stream_1_conv_1_layer_call_fn_3599260Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_3599305Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
л2и
1__inference_stream_2_conv_1_layer_call_fn_3599314Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2џ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3599334
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3599368
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3599388
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3599422Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
5__inference_batch_normalization_layer_call_fn_3599435
5__inference_batch_normalization_layer_call_fn_3599448
5__inference_batch_normalization_layer_call_fn_3599461
5__inference_batch_normalization_layer_call_fn_3599474Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3599494
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3599528
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3599548
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3599582Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
7__inference_batch_normalization_1_layer_call_fn_3599595
7__inference_batch_normalization_1_layer_call_fn_3599608
7__inference_batch_normalization_1_layer_call_fn_3599621
7__inference_batch_normalization_1_layer_call_fn_3599634Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3599654
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3599688
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3599708
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3599742Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
7__inference_batch_normalization_2_layer_call_fn_3599755
7__inference_batch_normalization_2_layer_call_fn_3599768
7__inference_batch_normalization_2_layer_call_fn_3599781
7__inference_batch_normalization_2_layer_call_fn_3599794Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ё2ю
G__inference_activation_layer_call_and_return_conditional_losses_3599799Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_activation_layer_call_fn_3599804Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_activation_1_layer_call_and_return_conditional_losses_3599809Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
и2е
.__inference_activation_1_layer_call_fn_3599814Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_activation_2_layer_call_and_return_conditional_losses_3599819Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
и2е
.__inference_activation_2_layer_call_fn_3599824Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ъ2Ч
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_3599832
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_3599840Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
4__inference_stream_0_maxpool_1_layer_call_fn_3599845
4__inference_stream_0_maxpool_1_layer_call_fn_3599850Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ъ2Ч
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_3599858
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_3599866Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
4__inference_stream_1_maxpool_1_layer_call_fn_3599871
4__inference_stream_1_maxpool_1_layer_call_fn_3599876Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ъ2Ч
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_3599884
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_3599892Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
4__inference_stream_2_maxpool_1_layer_call_fn_3599897
4__inference_stream_2_maxpool_1_layer_call_fn_3599902Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_3599907
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_3599919Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 2
1__inference_stream_0_drop_1_layer_call_fn_3599924
1__inference_stream_0_drop_1_layer_call_fn_3599929Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ж2г
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_3599934
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_3599946Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 2
1__inference_stream_1_drop_1_layer_call_fn_3599951
1__inference_stream_1_drop_1_layer_call_fn_3599956Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ж2г
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_3599961
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_3599973Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 2
1__inference_stream_2_drop_1_layer_call_fn_3599978
1__inference_stream_2_drop_1_layer_call_fn_3599983Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
у2р
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3599989
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3599995Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
­2Њ
:__inference_global_average_pooling1d_layer_call_fn_3600000
:__inference_global_average_pooling1d_layer_call_fn_3600005Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ч2ф
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_3600011
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_3600017Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Б2Ў
<__inference_global_average_pooling1d_1_layer_call_fn_3600022
<__inference_global_average_pooling1d_1_layer_call_fn_3600027Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ч2ф
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_3600033
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_3600039Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Б2Ў
<__inference_global_average_pooling1d_2_layer_call_fn_3600044
<__inference_global_average_pooling1d_2_layer_call_fn_3600049Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_concatenate_layer_call_and_return_conditional_losses_3600057Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_concatenate_layer_call_fn_3600064Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_3600069
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_3600073Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 2
1__inference_dense_1_dropout_layer_call_fn_3600078
1__inference_dense_1_dropout_layer_call_fn_3600083Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ю2ы
D__inference_dense_1_layer_call_and_return_conditional_losses_3600123Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_dense_1_layer_call_fn_3600132Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
т2п
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3600152
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3600186Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ќ2Љ
7__inference_batch_normalization_3_layer_call_fn_3600199
7__inference_batch_normalization_3_layer_call_fn_3600212Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
љ2і
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_3600216Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
о2л
4__inference_dense_activation_1_layer_call_fn_3600221Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Д2Б
__inference_loss_fn_0_3600241
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_1_3600261
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_2_3600281
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_3_3600301
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ Г
"__inference__wrapped_model_3594712-.+,)*>3=4<1;2:/9056@7?89Ђ6
/Ђ,
*'
left_inputsџџџџџџџџџњ
Њ "5Њ2
0
	basemodel# 
	basemodelџџџџџџџџџ@Џ
I__inference_activation_1_layer_call_and_return_conditional_losses_3599809b4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ@
Њ "*Ђ'
 
0џџџџџџџџџњ@
 
.__inference_activation_1_layer_call_fn_3599814U4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ@
Њ "џџџџџџџџџњ@Џ
I__inference_activation_2_layer_call_and_return_conditional_losses_3599819b4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ@
Њ "*Ђ'
 
0џџџџџџџџџњ@
 
.__inference_activation_2_layer_call_fn_3599824U4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ@
Њ "џџџџџџџџџњ@­
G__inference_activation_layer_call_and_return_conditional_losses_3599799b4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ@
Њ "*Ђ'
 
0џџџџџџџџџњ@
 
,__inference_activation_layer_call_fn_3599804U4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ@
Њ "џџџџџџџџџњ@Ї
F__inference_basemodel_layer_call_and_return_conditional_losses_3596888м-.+,)*>3=4<1;2:/9056@7?8Ђ
Ђ
~{
'$
inputs_0џџџџџџџџџњ
'$
inputs_1џџџџџџџџџњ
'$
inputs_2џџџџџџџџџњ
p 

 
Њ "%Ђ"

0џџџџџџџџџ@
 Ї
F__inference_basemodel_layer_call_and_return_conditional_losses_3597028м-.+,)*=>34;<129:/056?@78Ђ
Ђ
~{
'$
inputs_0џџџџџџџџџњ
'$
inputs_1џџџџџџџџџњ
'$
inputs_2џџџџџџџџџњ
p

 
Њ "%Ђ"

0џџџџџџџџџ@
 Ї
F__inference_basemodel_layer_call_and_return_conditional_losses_3598665м-.+,)*>3=4<1;2:/9056@7?8Ђ
Ђ
~{
'$
inputs/0џџџџџџџџџњ
'$
inputs/1џџџџџџџџџњ
'$
inputs/2џџџџџџџџџњ
p 

 
Њ "%Ђ"

0џџџџџџџџџ@
 Ї
F__inference_basemodel_layer_call_and_return_conditional_losses_3598961м-.+,)*=>34;<129:/056?@78Ђ
Ђ
~{
'$
inputs/0џџџџџџџџџњ
'$
inputs/1џџџџџџџџџњ
'$
inputs/2џџџџџџџџџњ
p

 
Њ "%Ђ"

0џџџџџџџџџ@
 џ
+__inference_basemodel_layer_call_fn_3596008Я-.+,)*>3=4<1;2:/9056@7?8Ђ
Ђ
~{
'$
inputs_0џџџџџџџџџњ
'$
inputs_1џџџџџџџџџњ
'$
inputs_2џџџџџџџџџњ
p 

 
Њ "џџџџџџџџџ@џ
+__inference_basemodel_layer_call_fn_3596748Я-.+,)*=>34;<129:/056?@78Ђ
Ђ
~{
'$
inputs_0џџџџџџџџџњ
'$
inputs_1џџџџџџџџџњ
'$
inputs_2џџџџџџџџџњ
p

 
Њ "џџџџџџџџџ@џ
+__inference_basemodel_layer_call_fn_3599016Я-.+,)*>3=4<1;2:/9056@7?8Ђ
Ђ
~{
'$
inputs/0џџџџџџџџџњ
'$
inputs/1џџџџџџџџџњ
'$
inputs/2џџџџџџџџџњ
p 

 
Њ "џџџџџџџџџ@џ
+__inference_basemodel_layer_call_fn_3599071Я-.+,)*=>34;<129:/056?@78Ђ
Ђ
~{
'$
inputs/0џџџџџџџџџњ
'$
inputs/1џџџџџџџџџњ
'$
inputs/2џџџџџџџџџњ
p

 
Њ "џџџџџџџџџ@в
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3599494|<1;2@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ@
p 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ@
 в
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3599528|;<12@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ@
p
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ@
 Т
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3599548l<1;28Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ@
p 
Њ "*Ђ'
 
0џџџџџџџџџњ@
 Т
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3599582l;<128Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ@
p
Њ "*Ђ'
 
0џџџџџџџџџњ@
 Њ
7__inference_batch_normalization_1_layer_call_fn_3599595o<1;2@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ@
p 
Њ "%"џџџџџџџџџџџџџџџџџџ@Њ
7__inference_batch_normalization_1_layer_call_fn_3599608o;<12@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ@
p
Њ "%"џџџџџџџџџџџџџџџџџџ@
7__inference_batch_normalization_1_layer_call_fn_3599621_<1;28Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ@
p 
Њ "џџџџџџџџџњ@
7__inference_batch_normalization_1_layer_call_fn_3599634_;<128Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ@
p
Њ "џџџџџџџџџњ@в
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3599654|>3=4@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ@
p 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ@
 в
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3599688|=>34@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ@
p
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ@
 Т
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3599708l>3=48Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ@
p 
Њ "*Ђ'
 
0џџџџџџџџџњ@
 Т
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3599742l=>348Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ@
p
Њ "*Ђ'
 
0џџџџџџџџџњ@
 Њ
7__inference_batch_normalization_2_layer_call_fn_3599755o>3=4@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ@
p 
Њ "%"џџџџџџџџџџџџџџџџџџ@Њ
7__inference_batch_normalization_2_layer_call_fn_3599768o=>34@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ@
p
Њ "%"џџџџџџџџџџџџџџџџџџ@
7__inference_batch_normalization_2_layer_call_fn_3599781_>3=48Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ@
p 
Њ "џџџџџџџџџњ@
7__inference_batch_normalization_2_layer_call_fn_3599794_=>348Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ@
p
Њ "џџџџџџџџџњ@И
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3600152b@7?83Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p 
Њ "%Ђ"

0џџџџџџџџџ@
 И
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3600186b?@783Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p
Њ "%Ђ"

0џџџџџџџџџ@
 
7__inference_batch_normalization_3_layer_call_fn_3600199U@7?83Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p 
Њ "џџџџџџџџџ@
7__inference_batch_normalization_3_layer_call_fn_3600212U?@783Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p
Њ "џџџџџџџџџ@а
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3599334|:/90@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ@
p 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ@
 а
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3599368|9:/0@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ@
p
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ@
 Р
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3599388l:/908Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ@
p 
Њ "*Ђ'
 
0џџџџџџџџџњ@
 Р
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3599422l9:/08Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ@
p
Њ "*Ђ'
 
0џџџџџџџџџњ@
 Ј
5__inference_batch_normalization_layer_call_fn_3599435o:/90@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ@
p 
Њ "%"џџџџџџџџџџџџџџџџџџ@Ј
5__inference_batch_normalization_layer_call_fn_3599448o9:/0@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџ@
p
Њ "%"џџџџџџџџџџџџџџџџџџ@
5__inference_batch_normalization_layer_call_fn_3599461_:/908Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ@
p 
Њ "џџџџџџџџџњ@
5__inference_batch_normalization_layer_call_fn_3599474_9:/08Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ@
p
Њ "џџџџџџџџџњ@ѕ
H__inference_concatenate_layer_call_and_return_conditional_losses_3600057Ј~Ђ{
tЂq
ol
"
inputs/0џџџџџџџџџ@
"
inputs/1џџџџџџџџџ@
"
inputs/2џџџџџџџџџ@
Њ "&Ђ#

0џџџџџџџџџР
 Э
-__inference_concatenate_layer_call_fn_3600064~Ђ{
tЂq
ol
"
inputs/0џџџџџџџџџ@
"
inputs/1џџџџџџџџџ@
"
inputs/2џџџџџџџџџ@
Њ "џџџџџџџџџРЎ
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_3600069^4Ђ1
*Ђ'
!
inputsџџџџџџџџџР
p 
Њ "&Ђ#

0џџџџџџџџџР
 Ў
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_3600073^4Ђ1
*Ђ'
!
inputsџџџџџџџџџР
p
Њ "&Ђ#

0џџџџџџџџџР
 
1__inference_dense_1_dropout_layer_call_fn_3600078Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџР
p 
Њ "џџџџџџџџџР
1__inference_dense_1_dropout_layer_call_fn_3600083Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџР
p
Њ "џџџџџџџџџРЅ
D__inference_dense_1_layer_call_and_return_conditional_losses_3600123]560Ђ-
&Ђ#
!
inputsџџџџџџџџџР
Њ "%Ђ"

0џџџџџџџџџ@
 }
)__inference_dense_1_layer_call_fn_3600132P560Ђ-
&Ђ#
!
inputsџџџџџџџџџР
Њ "џџџџџџџџџ@Ћ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_3600216X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ@
 
4__inference_dense_activation_1_layer_call_fn_3600221K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@ж
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_3600011{IЂF
?Ђ<
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 Л
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_3600017`7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ}@

 
Њ "%Ђ"

0џџџџџџџџџ@
 Ў
<__inference_global_average_pooling1d_1_layer_call_fn_3600022nIЂF
?Ђ<
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
Њ "!џџџџџџџџџџџџџџџџџџ
<__inference_global_average_pooling1d_1_layer_call_fn_3600027S7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ}@

 
Њ "џџџџџџџџџ@ж
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_3600033{IЂF
?Ђ<
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 Л
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_3600039`7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ}@

 
Њ "%Ђ"

0џџџџџџџџџ@
 Ў
<__inference_global_average_pooling1d_2_layer_call_fn_3600044nIЂF
?Ђ<
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
Њ "!џџџџџџџџџџџџџџџџџџ
<__inference_global_average_pooling1d_2_layer_call_fn_3600049S7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ}@

 
Њ "џџџџџџџџџ@д
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3599989{IЂF
?Ђ<
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 Й
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3599995`7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ}@

 
Њ "%Ђ"

0џџџџџџџџџ@
 Ќ
:__inference_global_average_pooling1d_layer_call_fn_3600000nIЂF
?Ђ<
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
Њ "!џџџџџџџџџџџџџџџџџџ
:__inference_global_average_pooling1d_layer_call_fn_3600005S7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ}@

 
Њ "џџџџџџџџџ@<
__inference_loss_fn_0_3600241)Ђ

Ђ 
Њ " <
__inference_loss_fn_1_3600261+Ђ

Ђ 
Њ " <
__inference_loss_fn_2_3600281-Ђ

Ђ 
Њ " <
__inference_loss_fn_3_36003015Ђ

Ђ 
Њ " Э
D__inference_model_1_layer_call_and_return_conditional_losses_3597581-.+,)*>3=4<1;2:/9056@7?8AЂ>
7Ђ4
*'
left_inputsџџџџџџџџџњ
p 

 
Њ "%Ђ"

0џџџџџџџџџ@
 Э
D__inference_model_1_layer_call_and_return_conditional_losses_3597694-.+,)*=>34;<129:/056?@78AЂ>
7Ђ4
*'
left_inputsџџџџџџџџџњ
p

 
Њ "%Ђ"

0џџџџџџџџџ@
 Ч
D__inference_model_1_layer_call_and_return_conditional_losses_3598006-.+,)*>3=4<1;2:/9056@7?8<Ђ9
2Ђ/
%"
inputsџџџџџџџџџњ
p 

 
Њ "%Ђ"

0џџџџџџџџџ@
 Ч
D__inference_model_1_layer_call_and_return_conditional_losses_3598300-.+,)*=>34;<129:/056?@78<Ђ9
2Ђ/
%"
inputsџџџџџџџџџњ
p

 
Њ "%Ђ"

0џџџџџџџџџ@
 Є
)__inference_model_1_layer_call_fn_3597196w-.+,)*>3=4<1;2:/9056@7?8AЂ>
7Ђ4
*'
left_inputsџџџџџџџџџњ
p 

 
Њ "џџџџџџџџџ@Є
)__inference_model_1_layer_call_fn_3597468w-.+,)*=>34;<129:/056?@78AЂ>
7Ђ4
*'
left_inputsџџџџџџџџџњ
p

 
Њ "џџџџџџџџџ@
)__inference_model_1_layer_call_fn_3598353r-.+,)*>3=4<1;2:/9056@7?8<Ђ9
2Ђ/
%"
inputsџџџџџџџџџњ
p 

 
Њ "џџџџџџџџџ@
)__inference_model_1_layer_call_fn_3598406r-.+,)*=>34;<129:/056?@78<Ђ9
2Ђ/
%"
inputsџџџџџџџџџњ
p

 
Њ "џџџџџџџџџ@Х
%__inference_signature_wrapper_3597809-.+,)*>3=4<1;2:/9056@7?8HЂE
Ђ 
>Њ;
9
left_inputs*'
left_inputsџџџџџџџџџњ"5Њ2
0
	basemodel# 
	basemodelџџџџџџџџџ@Ж
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_3599197f)*4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ
Њ "*Ђ'
 
0џџџџџџџџџњ@
 
1__inference_stream_0_conv_1_layer_call_fn_3599206Y)*4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ
Њ "џџџџџџџџџњ@Д
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_3599907d7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ}@
p 
Њ ")Ђ&

0џџџџџџџџџ}@
 Д
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_3599919d7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ}@
p
Њ ")Ђ&

0џџџџџџџџџ}@
 
1__inference_stream_0_drop_1_layer_call_fn_3599924W7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ}@
p 
Њ "џџџџџџџџџ}@
1__inference_stream_0_drop_1_layer_call_fn_3599929W7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ}@
p
Њ "џџџџџџџџџ}@К
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_3599076f8Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ
p 
Њ "*Ђ'
 
0џџџџџџџџџњ
 К
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_3599088f8Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ
p
Њ "*Ђ'
 
0џџџџџџџџџњ
 
5__inference_stream_0_input_drop_layer_call_fn_3599093Y8Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ
p 
Њ "џџџџџџџџџњ
5__inference_stream_0_input_drop_layer_call_fn_3599098Y8Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ
p
Њ "џџџџџџџџџњи
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_3599832EЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";Ђ8
1.
0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Д
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_3599840a4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ@
Њ ")Ђ&

0џџџџџџџџџ}@
 Џ
4__inference_stream_0_maxpool_1_layer_call_fn_3599845wEЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
4__inference_stream_0_maxpool_1_layer_call_fn_3599850T4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ@
Њ "џџџџџџџџџ}@Ж
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_3599251f+,4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ
Њ "*Ђ'
 
0џџџџџџџџџњ@
 
1__inference_stream_1_conv_1_layer_call_fn_3599260Y+,4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ
Њ "џџџџџџџџџњ@Д
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_3599934d7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ}@
p 
Њ ")Ђ&

0џџџџџџџџџ}@
 Д
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_3599946d7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ}@
p
Њ ")Ђ&

0џџџџџџџџџ}@
 
1__inference_stream_1_drop_1_layer_call_fn_3599951W7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ}@
p 
Њ "џџџџџџџџџ}@
1__inference_stream_1_drop_1_layer_call_fn_3599956W7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ}@
p
Њ "џџџџџџџџџ}@К
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_3599103f8Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ
p 
Њ "*Ђ'
 
0џџџџџџџџџњ
 К
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_3599115f8Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ
p
Њ "*Ђ'
 
0џџџџџџџџџњ
 
5__inference_stream_1_input_drop_layer_call_fn_3599120Y8Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ
p 
Њ "џџџџџџџџџњ
5__inference_stream_1_input_drop_layer_call_fn_3599125Y8Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ
p
Њ "џџџџџџџџџњи
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_3599858EЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";Ђ8
1.
0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Д
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_3599866a4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ@
Њ ")Ђ&

0џџџџџџџџџ}@
 Џ
4__inference_stream_1_maxpool_1_layer_call_fn_3599871wEЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
4__inference_stream_1_maxpool_1_layer_call_fn_3599876T4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ@
Њ "џџџџџџџџџ}@Ж
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_3599305f-.4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ
Њ "*Ђ'
 
0џџџџџџџџџњ@
 
1__inference_stream_2_conv_1_layer_call_fn_3599314Y-.4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ
Њ "џџџџџџџџџњ@Д
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_3599961d7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ}@
p 
Њ ")Ђ&

0џџџџџџџџџ}@
 Д
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_3599973d7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ}@
p
Њ ")Ђ&

0џџџџџџџџџ}@
 
1__inference_stream_2_drop_1_layer_call_fn_3599978W7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ}@
p 
Њ "џџџџџџџџџ}@
1__inference_stream_2_drop_1_layer_call_fn_3599983W7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ}@
p
Њ "џџџџџџџџџ}@К
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_3599130f8Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ
p 
Њ "*Ђ'
 
0џџџџџџџџџњ
 К
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_3599142f8Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ
p
Њ "*Ђ'
 
0џџџџџџџџџњ
 
5__inference_stream_2_input_drop_layer_call_fn_3599147Y8Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ
p 
Њ "џџџџџџџџџњ
5__inference_stream_2_input_drop_layer_call_fn_3599152Y8Ђ5
.Ђ+
%"
inputsџџџџџџџџџњ
p
Њ "џџџџџџџџџњи
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_3599884EЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";Ђ8
1.
0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Д
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_3599892a4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ@
Њ ")Ђ&

0џџџџџџџџџ}@
 Џ
4__inference_stream_2_maxpool_1_layer_call_fn_3599897wEЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
4__inference_stream_2_maxpool_1_layer_call_fn_3599902T4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ@
Њ "џџџџџџџџџ}@