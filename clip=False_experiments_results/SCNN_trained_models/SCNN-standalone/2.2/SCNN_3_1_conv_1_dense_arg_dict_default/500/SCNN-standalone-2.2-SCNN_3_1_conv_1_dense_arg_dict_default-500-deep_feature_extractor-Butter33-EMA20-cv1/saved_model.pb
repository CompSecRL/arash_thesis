зЖ4
–¶
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
 И"serve*2.6.22v2.6.1-9-gc2363d6d0258≈о/
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
shape:	ј@*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	ј@*
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
Йc
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ƒb
valueЇbBЈb B∞b
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
Ѓ
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
%trainable_variables
&	variables
'regularization_losses
(	keras_api
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
ґ
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
 
≠

Alayers
trainable_variables
Blayer_regularization_losses
Cmetrics
	variables
regularization_losses
Dnon_trainable_variables
Elayer_metrics
 
 
 
 
R
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
R
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
R
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
h

)kernel
*bias
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
h

+kernel
,bias
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
h

-kernel
.bias
Ztrainable_variables
[	variables
\regularization_losses
]	keras_api
Ч
^axis
	/gamma
0beta
9moving_mean
:moving_variance
_trainable_variables
`	variables
aregularization_losses
b	keras_api
Ч
caxis
	1gamma
2beta
;moving_mean
<moving_variance
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
Ч
haxis
	3gamma
4beta
=moving_mean
>moving_variance
itrainable_variables
j	variables
kregularization_losses
l	keras_api
R
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
R
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
R
utrainable_variables
v	variables
wregularization_losses
x	keras_api
R
ytrainable_variables
z	variables
{regularization_losses
|	keras_api
S
}trainable_variables
~	variables
regularization_losses
А	keras_api
V
Бtrainable_variables
В	variables
Гregularization_losses
Д	keras_api
V
Еtrainable_variables
Ж	variables
Зregularization_losses
И	keras_api
V
Йtrainable_variables
К	variables
Лregularization_losses
М	keras_api
V
Нtrainable_variables
О	variables
Пregularization_losses
Р	keras_api
V
Сtrainable_variables
Т	variables
Уregularization_losses
Ф	keras_api
V
Хtrainable_variables
Ц	variables
Чregularization_losses
Ш	keras_api
V
Щtrainable_variables
Ъ	variables
Ыregularization_losses
Ь	keras_api
V
Эtrainable_variables
Ю	variables
Яregularization_losses
†	keras_api
V
°trainable_variables
Ґ	variables
£regularization_losses
§	keras_api
l

5kernel
6bias
•trainable_variables
¶	variables
Іregularization_losses
®	keras_api
Ь
	©axis
	7gamma
8beta
?moving_mean
@moving_variance
™trainable_variables
Ђ	variables
ђregularization_losses
≠	keras_api
V
Ѓtrainable_variables
ѓ	variables
∞regularization_losses
±	keras_api
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
ґ
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
 
≤
≤layers
%trainable_variables
 ≥layer_regularization_losses
іmetrics
&	variables
'regularization_losses
µnon_trainable_variables
ґlayer_metrics
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

0
1
 
 
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
 
≤
Јlayers
Ftrainable_variables
 Єlayer_regularization_losses
єmetrics
G	variables
Hregularization_losses
Їnon_trainable_variables
їlayer_metrics
 
 
 
≤
Љlayers
Jtrainable_variables
 љlayer_regularization_losses
Њmetrics
K	variables
Lregularization_losses
њnon_trainable_variables
јlayer_metrics
 
 
 
≤
Ѕlayers
Ntrainable_variables
 ¬layer_regularization_losses
√metrics
O	variables
Pregularization_losses
ƒnon_trainable_variables
≈layer_metrics

)0
*1

)0
*1
 
≤
∆layers
Rtrainable_variables
 «layer_regularization_losses
»metrics
S	variables
Tregularization_losses
…non_trainable_variables
 layer_metrics

+0
,1

+0
,1
 
≤
Ћlayers
Vtrainable_variables
 ћlayer_regularization_losses
Ќmetrics
W	variables
Xregularization_losses
ќnon_trainable_variables
ѕlayer_metrics

-0
.1

-0
.1
 
≤
–layers
Ztrainable_variables
 —layer_regularization_losses
“metrics
[	variables
\regularization_losses
”non_trainable_variables
‘layer_metrics
 

/0
01

/0
01
92
:3
 
≤
’layers
_trainable_variables
 ÷layer_regularization_losses
„metrics
`	variables
aregularization_losses
Ўnon_trainable_variables
ўlayer_metrics
 

10
21

10
21
;2
<3
 
≤
Џlayers
dtrainable_variables
 џlayer_regularization_losses
№metrics
e	variables
fregularization_losses
Ёnon_trainable_variables
ёlayer_metrics
 

30
41

30
41
=2
>3
 
≤
яlayers
itrainable_variables
 аlayer_regularization_losses
бmetrics
j	variables
kregularization_losses
вnon_trainable_variables
гlayer_metrics
 
 
 
≤
дlayers
mtrainable_variables
 еlayer_regularization_losses
жmetrics
n	variables
oregularization_losses
зnon_trainable_variables
иlayer_metrics
 
 
 
≤
йlayers
qtrainable_variables
 кlayer_regularization_losses
лmetrics
r	variables
sregularization_losses
мnon_trainable_variables
нlayer_metrics
 
 
 
≤
оlayers
utrainable_variables
 пlayer_regularization_losses
рmetrics
v	variables
wregularization_losses
сnon_trainable_variables
тlayer_metrics
 
 
 
≤
уlayers
ytrainable_variables
 фlayer_regularization_losses
хmetrics
z	variables
{regularization_losses
цnon_trainable_variables
чlayer_metrics
 
 
 
≤
шlayers
}trainable_variables
 щlayer_regularization_losses
ъmetrics
~	variables
regularization_losses
ыnon_trainable_variables
ьlayer_metrics
 
 
 
µ
эlayers
Бtrainable_variables
 юlayer_regularization_losses
€metrics
В	variables
Гregularization_losses
Аnon_trainable_variables
Бlayer_metrics
 
 
 
µ
Вlayers
Еtrainable_variables
 Гlayer_regularization_losses
Дmetrics
Ж	variables
Зregularization_losses
Еnon_trainable_variables
Жlayer_metrics
 
 
 
µ
Зlayers
Йtrainable_variables
 Иlayer_regularization_losses
Йmetrics
К	variables
Лregularization_losses
Кnon_trainable_variables
Лlayer_metrics
 
 
 
µ
Мlayers
Нtrainable_variables
 Нlayer_regularization_losses
Оmetrics
О	variables
Пregularization_losses
Пnon_trainable_variables
Рlayer_metrics
 
 
 
µ
Сlayers
Сtrainable_variables
 Тlayer_regularization_losses
Уmetrics
Т	variables
Уregularization_losses
Фnon_trainable_variables
Хlayer_metrics
 
 
 
µ
Цlayers
Хtrainable_variables
 Чlayer_regularization_losses
Шmetrics
Ц	variables
Чregularization_losses
Щnon_trainable_variables
Ъlayer_metrics
 
 
 
µ
Ыlayers
Щtrainable_variables
 Ьlayer_regularization_losses
Эmetrics
Ъ	variables
Ыregularization_losses
Юnon_trainable_variables
Яlayer_metrics
 
 
 
µ
†layers
Эtrainable_variables
 °layer_regularization_losses
Ґmetrics
Ю	variables
Яregularization_losses
£non_trainable_variables
§layer_metrics
 
 
 
µ
•layers
°trainable_variables
 ¶layer_regularization_losses
Іmetrics
Ґ	variables
£regularization_losses
®non_trainable_variables
©layer_metrics

50
61

50
61
 
µ
™layers
•trainable_variables
 Ђlayer_regularization_losses
ђmetrics
¶	variables
Іregularization_losses
≠non_trainable_variables
Ѓlayer_metrics
 

70
81

70
81
?2
@3
 
µ
ѓlayers
™trainable_variables
 ∞layer_regularization_losses
±metrics
Ђ	variables
ђregularization_losses
≤non_trainable_variables
≥layer_metrics
 
 
 
µ
іlayers
Ѓtrainable_variables
 µlayer_regularization_losses
ґmetrics
ѓ	variables
∞regularization_losses
Јnon_trainable_variables
Єlayer_metrics
ё
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
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
И
serving_default_left_inputsPlaceholder*,
_output_shapes
:€€€€€€€€€ф*
dtype0*!
shape:€€€€€€€€€ф
Ц
StatefulPartitionedCallStatefulPartitionedCallserving_default_left_inputsstream_2_conv_1/kernelstream_2_conv_1/biasstream_1_conv_1/kernelstream_1_conv_1/biasstream_0_conv_1/kernelstream_0_conv_1/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/beta%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/beta#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense_1/kerneldense_1/bias%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/beta*$
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
%__inference_signature_wrapper_2260645
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
√
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
GPU2*0J 8В *)
f$R"
 __inference__traced_save_2263232
ё
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
GPU2*0J 8В *,
f'R%
#__inference__traced_restore_2263314щЄ.
Ј+
й
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2257632

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
Є
±
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2262542

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
ц
±
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2263014

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
Ц
k
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_2258046

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
С
n
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_2258368

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
є+
л
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2262576

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
и
В
H__inference_concatenate_layer_call_and_return_conditional_losses_2262900
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
ФК
Ж
F__inference_basemodel_layer_call_and_return_conditional_losses_2261907
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
&dense_1_matmul_readvariableop_resource:	ј@5
'dense_1_biasadd_readvariableop_resource:@K
=batch_normalization_3_assignmovingavg_readvariableop_resource:@M
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_3_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_3_batchnorm_readvariableop_resource:@
identityИҐ#batch_normalization/AssignMovingAvgҐ2batch_normalization/AssignMovingAvg/ReadVariableOpҐ%batch_normalization/AssignMovingAvg_1Ґ4batch_normalization/AssignMovingAvg_1/ReadVariableOpҐ,batch_normalization/batchnorm/ReadVariableOpҐ0batch_normalization/batchnorm/mul/ReadVariableOpҐ%batch_normalization_1/AssignMovingAvgҐ4batch_normalization_1/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_1/AssignMovingAvg_1Ґ6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_1/batchnorm/ReadVariableOpҐ2batch_normalization_1/batchnorm/mul/ReadVariableOpҐ%batch_normalization_2/AssignMovingAvgҐ4batch_normalization_2/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_2/AssignMovingAvg_1Ґ6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_2/batchnorm/ReadVariableOpҐ2batch_normalization_2/batchnorm/mul/ReadVariableOpҐ%batch_normalization_3/AssignMovingAvgҐ4batch_normalization_3/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_3/AssignMovingAvg_1Ґ6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_3/batchnorm/ReadVariableOpҐ2batch_normalization_3/batchnorm/mul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_1/BiasAdd/ReadVariableOpҐ2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_1_conv_1/BiasAdd/ReadVariableOpҐ2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_2_conv_1/BiasAdd/ReadVariableOpҐ2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЛ
!stream_2_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2#
!stream_2_input_drop/dropout/Constґ
stream_2_input_drop/dropout/MulMulinputs_2*stream_2_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2!
stream_2_input_drop/dropout/Mul~
!stream_2_input_drop/dropout/ShapeShapeinputs_2*
T0*
_output_shapes
:2#
!stream_2_input_drop/dropout/ShapeР
8stream_2_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_2_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф*
dtype0*
seedЈ*
seed2є2:
8stream_2_input_drop/dropout/random_uniform/RandomUniformЭ
*stream_2_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2,
*stream_2_input_drop/dropout/GreaterEqual/yУ
(stream_2_input_drop/dropout/GreaterEqualGreaterEqualAstream_2_input_drop/dropout/random_uniform/RandomUniform:output:03stream_2_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2*
(stream_2_input_drop/dropout/GreaterEqualј
 stream_2_input_drop/dropout/CastCast,stream_2_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ф2"
 stream_2_input_drop/dropout/Castѕ
!stream_2_input_drop/dropout/Mul_1Mul#stream_2_input_drop/dropout/Mul:z:0$stream_2_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ф2#
!stream_2_input_drop/dropout/Mul_1Л
!stream_1_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2#
!stream_1_input_drop/dropout/Constґ
stream_1_input_drop/dropout/MulMulinputs_1*stream_1_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2!
stream_1_input_drop/dropout/Mul~
!stream_1_input_drop/dropout/ShapeShapeinputs_1*
T0*
_output_shapes
:2#
!stream_1_input_drop/dropout/ShapeР
8stream_1_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_1_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф*
dtype0*
seedЈ*
seed2Є2:
8stream_1_input_drop/dropout/random_uniform/RandomUniformЭ
*stream_1_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2,
*stream_1_input_drop/dropout/GreaterEqual/yУ
(stream_1_input_drop/dropout/GreaterEqualGreaterEqualAstream_1_input_drop/dropout/random_uniform/RandomUniform:output:03stream_1_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2*
(stream_1_input_drop/dropout/GreaterEqualј
 stream_1_input_drop/dropout/CastCast,stream_1_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ф2"
 stream_1_input_drop/dropout/Castѕ
!stream_1_input_drop/dropout/Mul_1Mul#stream_1_input_drop/dropout/Mul:z:0$stream_1_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ф2#
!stream_1_input_drop/dropout/Mul_1Л
!stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2#
!stream_0_input_drop/dropout/Constґ
stream_0_input_drop/dropout/MulMulinputs_0*stream_0_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2!
stream_0_input_drop/dropout/Mul~
!stream_0_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/ShapeР
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф*
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
*stream_0_input_drop/dropout/GreaterEqual/yУ
(stream_0_input_drop/dropout/GreaterEqualGreaterEqualAstream_0_input_drop/dropout/random_uniform/RandomUniform:output:03stream_0_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2*
(stream_0_input_drop/dropout/GreaterEqualј
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ф2"
 stream_0_input_drop/dropout/Castѕ
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ф2#
!stream_0_input_drop/dropout/Mul_1Щ
%stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_2_conv_1/conv1d/ExpandDims/dimж
!stream_2_conv_1/conv1d/ExpandDims
ExpandDims%stream_2_input_drop/dropout/Mul_1:z:0.stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф2#
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
#stream_2_conv_1/conv1d/ExpandDims_1ч
stream_2_conv_1/conv1dConv2D*stream_2_conv_1/conv1d/ExpandDims:output:0,stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2
stream_2_conv_1/conv1d√
stream_2_conv_1/conv1d/SqueezeSqueezestream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€2 
stream_2_conv_1/conv1d/SqueezeЉ
&stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_2_conv_1/BiasAdd/ReadVariableOpЌ
stream_2_conv_1/BiasAddBiasAdd'stream_2_conv_1/conv1d/Squeeze:output:0.stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
stream_2_conv_1/BiasAddЩ
%stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_1_conv_1/conv1d/ExpandDims/dimж
!stream_1_conv_1/conv1d/ExpandDims
ExpandDims%stream_1_input_drop/dropout/Mul_1:z:0.stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф2#
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
#stream_1_conv_1/conv1d/ExpandDims_1ч
stream_1_conv_1/conv1dConv2D*stream_1_conv_1/conv1d/ExpandDims:output:0,stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2
stream_1_conv_1/conv1d√
stream_1_conv_1/conv1d/SqueezeSqueezestream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€2 
stream_1_conv_1/conv1d/SqueezeЉ
&stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_1_conv_1/BiasAdd/ReadVariableOpЌ
stream_1_conv_1/BiasAddBiasAdd'stream_1_conv_1/conv1d/Squeeze:output:0.stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
stream_1_conv_1/BiasAddЩ
%stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_0_conv_1/conv1d/ExpandDims/dimж
!stream_0_conv_1/conv1d/ExpandDims
ExpandDims%stream_0_input_drop/dropout/Mul_1:z:0.stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф2#
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
#stream_0_conv_1/conv1d/ExpandDims_1ч
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d√
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€2 
stream_0_conv_1/conv1d/SqueezeЉ
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOpЌ
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
*batch_normalization_2/moments/StopGradientЕ
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference stream_2_conv_1/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@21
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
#batch_normalization_2/batchnorm/mul„
%batch_normalization_2/batchnorm/mul_1Mul stream_2_conv_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2'
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
#batch_normalization_2/batchnorm/subв
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2'
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
*batch_normalization_1/moments/StopGradientЕ
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_1_conv_1/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@21
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
#batch_normalization_1/batchnorm/mul„
%batch_normalization_1/batchnorm/mul_1Mul stream_1_conv_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2'
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
#batch_normalization_1/batchnorm/subв
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2'
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
(batch_normalization/moments/StopGradient€
-batch_normalization/moments/SquaredDifferenceSquaredDifference stream_0_conv_1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2/
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
!batch_normalization/batchnorm/mul—
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2%
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
!batch_normalization/batchnorm/subЏ
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2%
#batch_normalization/batchnorm/add_1Р
activation_2/TanhTanh)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
activation_2/TanhР
activation_1/TanhTanh)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
activation_1/TanhК
activation/TanhTanh'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
activation/TanhИ
!stream_2_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_2_maxpool_1/ExpandDims/dim 
stream_2_maxpool_1/ExpandDims
ExpandDimsactivation_2/Tanh:y:0*stream_2_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@2
stream_2_maxpool_1/ExpandDimsў
stream_2_maxpool_1/MaxPoolMaxPool&stream_2_maxpool_1/ExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2
stream_2_maxpool_1/MaxPoolґ
stream_2_maxpool_1/SqueezeSqueeze#stream_2_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2
stream_2_maxpool_1/SqueezeИ
!stream_1_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_1_maxpool_1/ExpandDims/dim 
stream_1_maxpool_1/ExpandDims
ExpandDimsactivation_1/Tanh:y:0*stream_1_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@2
stream_1_maxpool_1/ExpandDimsў
stream_1_maxpool_1/MaxPoolMaxPool&stream_1_maxpool_1/ExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2
stream_1_maxpool_1/MaxPoolґ
stream_1_maxpool_1/SqueezeSqueeze#stream_1_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2
stream_1_maxpool_1/SqueezeИ
!stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_1/ExpandDims/dim»
stream_0_maxpool_1/ExpandDims
ExpandDimsactivation/Tanh:y:0*stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@2
stream_0_maxpool_1/ExpandDimsў
stream_0_maxpool_1/MaxPoolMaxPool&stream_0_maxpool_1/ExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_1/MaxPoolґ
stream_0_maxpool_1/SqueezeSqueeze#stream_0_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2
stream_0_maxpool_1/SqueezeГ
stream_2_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
stream_2_drop_1/dropout/Const≈
stream_2_drop_1/dropout/MulMul#stream_2_maxpool_1/Squeeze:output:0&stream_2_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
stream_2_drop_1/dropout/MulС
stream_2_drop_1/dropout/ShapeShape#stream_2_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2
stream_2_drop_1/dropout/ShapeД
4stream_2_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_2_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
dtype0*
seedЈ*
seed2Ї26
4stream_2_drop_1/dropout/random_uniform/RandomUniformХ
&stream_2_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2(
&stream_2_drop_1/dropout/GreaterEqual/yГ
$stream_2_drop_1/dropout/GreaterEqualGreaterEqual=stream_2_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_2_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2&
$stream_2_drop_1/dropout/GreaterEqualі
stream_2_drop_1/dropout/CastCast(stream_2_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ъ@2
stream_2_drop_1/dropout/Castњ
stream_2_drop_1/dropout/Mul_1Mulstream_2_drop_1/dropout/Mul:z:0 stream_2_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
stream_2_drop_1/dropout/Mul_1Г
stream_1_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
stream_1_drop_1/dropout/Const≈
stream_1_drop_1/dropout/MulMul#stream_1_maxpool_1/Squeeze:output:0&stream_1_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
stream_1_drop_1/dropout/MulС
stream_1_drop_1/dropout/ShapeShape#stream_1_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2
stream_1_drop_1/dropout/ShapeД
4stream_1_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_1_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
dtype0*
seedЈ*
seed2є26
4stream_1_drop_1/dropout/random_uniform/RandomUniformХ
&stream_1_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2(
&stream_1_drop_1/dropout/GreaterEqual/yГ
$stream_1_drop_1/dropout/GreaterEqualGreaterEqual=stream_1_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_1_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2&
$stream_1_drop_1/dropout/GreaterEqualі
stream_1_drop_1/dropout/CastCast(stream_1_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ъ@2
stream_1_drop_1/dropout/Castњ
stream_1_drop_1/dropout/Mul_1Mulstream_1_drop_1/dropout/Mul:z:0 stream_1_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
stream_1_drop_1/dropout/Mul_1Г
stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
stream_0_drop_1/dropout/Const≈
stream_0_drop_1/dropout/MulMul#stream_0_maxpool_1/Squeeze:output:0&stream_0_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
stream_0_drop_1/dropout/MulС
stream_0_drop_1/dropout/ShapeShape#stream_0_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/ShapeД
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
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
&stream_0_drop_1/dropout/GreaterEqual/yГ
$stream_0_drop_1/dropout/GreaterEqualGreaterEqual=stream_0_drop_1/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2&
$stream_0_drop_1/dropout/GreaterEqualі
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ъ@2
stream_0_drop_1/dropout/Castњ
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
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
:	ј@*
dtype02
dense_1/MatMul/ReadVariableOp†
dense_1/MatMulMatMulconcatenate/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
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
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Constо
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs≠
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1ў
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_1_conv_1/kernel/Regularizer/mulў
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addф
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square≠
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2а
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xд
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1Ў
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1Щ
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Constо
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs≠
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1ў
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_2_conv_1/kernel/Regularizer/mulў
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addф
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square≠
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2а
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xд
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1Ў
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const∆
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ј@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2 
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
dense_1/kernel/Regularizer/addћ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ј@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2#
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
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_1_conv_1/BiasAdd/ReadVariableOp3^stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_2_conv_1/BiasAdd/ReadVariableOp3^stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:€€€€€€€€€ф:€€€€€€€€€ф:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 2J
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
:€€€€€€€€€ф
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs/2
£
X
<__inference_global_average_pooling1d_2_layer_call_fn_2262868

inputs
identityб
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
GPU2*0J 8В *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_22581762
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
ѓ
k
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_2258616

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimВ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@2

ExpandDims†
MaxPoolMaxPoolExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2	
MaxPool}
SqueezeSqueezeMaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2	
Squeezei
IdentityIdentitySqueeze:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф@:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
И+
л
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2259109

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
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
:€€€€€€€€€ф@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ф@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
С
n
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_2261976

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Э
Ґ
1__inference_stream_0_conv_1_layer_call_fn_2262012

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_22584862
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ф: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Ё
J
.__inference_activation_1_layer_call_fn_2262645

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_22585912
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф@:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
л
e
I__inference_activation_1_layer_call_and_return_conditional_losses_2262650

inputs
identityS
TanhTanhinputs*
T0*,
_output_shapes
:€€€€€€€€€ф@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф@:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
п
“
7__inference_batch_normalization_1_layer_call_fn_2262362

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
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22591092
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ф@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
П{
э

D__inference_model_1_layer_call_and_return_conditional_losses_2260530
left_inputs'
basemodel_2260420:@
basemodel_2260422:@'
basemodel_2260424:@
basemodel_2260426:@'
basemodel_2260428:@
basemodel_2260430:@
basemodel_2260432:@
basemodel_2260434:@
basemodel_2260436:@
basemodel_2260438:@
basemodel_2260440:@
basemodel_2260442:@
basemodel_2260444:@
basemodel_2260446:@
basemodel_2260448:@
basemodel_2260450:@
basemodel_2260452:@
basemodel_2260454:@$
basemodel_2260456:	ј@
basemodel_2260458:@
basemodel_2260460:@
basemodel_2260462:@
basemodel_2260464:@
basemodel_2260466:@
identityИҐ!basemodel/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЖ
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsleft_inputsleft_inputsbasemodel_2260420basemodel_2260422basemodel_2260424basemodel_2260426basemodel_2260428basemodel_2260430basemodel_2260432basemodel_2260434basemodel_2260436basemodel_2260438basemodel_2260440basemodel_2260442basemodel_2260444basemodel_2260446basemodel_2260448basemodel_2260450basemodel_2260452basemodel_2260454basemodel_2260456basemodel_2260458basemodel_2260460basemodel_2260462basemodel_2260464basemodel_2260466*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_22594782#
!basemodel/StatefulPartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constƒ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_2260428*"
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
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_2260428*"
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
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Constƒ
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_2260424*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs≠
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1ў
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_1_conv_1/kernel/Regularizer/mulў
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/add 
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_2260424*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square≠
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2а
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xд
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1Ў
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1Щ
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Constƒ
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_2260420*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs≠
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1ў
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_2_conv_1/kernel/Regularizer/mulў
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/add 
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_2260420*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square≠
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2а
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xд
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1Ў
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const±
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_2260456*
_output_shapes
:	ј@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2 
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
dense_1/kernel/Regularizer/addЈ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_2260456*
_output_shapes
:	ј@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2#
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
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 2F
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
:€€€€€€€€€ф
%
_user_specified_nameleft_inputs
Ип
≤
D__inference_model_1_layer_call_and_return_conditional_losses_2260948

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
0basemodel_dense_1_matmul_readvariableop_resource:	ј@?
1basemodel_dense_1_biasadd_readvariableop_resource:@O
Abasemodel_batch_normalization_3_batchnorm_readvariableop_resource:@S
Ebasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:@Q
Cbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource:@Q
Cbasemodel_batch_normalization_3_batchnorm_readvariableop_2_resource:@
identityИҐ6basemodel/batch_normalization/batchnorm/ReadVariableOpҐ8basemodel/batch_normalization/batchnorm/ReadVariableOp_1Ґ8basemodel/batch_normalization/batchnorm/ReadVariableOp_2Ґ:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpҐ8basemodel/batch_normalization_1/batchnorm/ReadVariableOpҐ:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1Ґ:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2Ґ<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpҐ8basemodel/batch_normalization_2/batchnorm/ReadVariableOpҐ:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1Ґ:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2Ґ<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpҐ8basemodel/batch_normalization_3/batchnorm/ReadVariableOpҐ:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1Ґ:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2Ґ<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpҐ(basemodel/dense_1/BiasAdd/ReadVariableOpҐ'basemodel/dense_1/MatMul/ReadVariableOpҐ0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpҐ<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpҐ<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpҐ<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЫ
&basemodel/stream_2_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ф2(
&basemodel/stream_2_input_drop/IdentityЫ
&basemodel/stream_1_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ф2(
&basemodel/stream_1_input_drop/IdentityЫ
&basemodel/stream_0_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ф2(
&basemodel/stream_0_input_drop/Identity≠
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€21
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimО
+basemodel/stream_2_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_2_input_drop/Identity:output:08basemodel/stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф2-
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
-basemodel/stream_2_conv_1/conv1d/ExpandDims_1Я
 basemodel/stream_2_conv_1/conv1dConv2D4basemodel/stream_2_conv_1/conv1d/ExpandDims:output:06basemodel/stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2"
 basemodel/stream_2_conv_1/conv1dб
(basemodel/stream_2_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€2*
(basemodel/stream_2_conv_1/conv1d/SqueezeЏ
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpх
!basemodel/stream_2_conv_1/BiasAddBiasAdd1basemodel/stream_2_conv_1/conv1d/Squeeze:output:08basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2#
!basemodel/stream_2_conv_1/BiasAdd≠
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€21
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimО
+basemodel/stream_1_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_1_input_drop/Identity:output:08basemodel/stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф2-
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
-basemodel/stream_1_conv_1/conv1d/ExpandDims_1Я
 basemodel/stream_1_conv_1/conv1dConv2D4basemodel/stream_1_conv_1/conv1d/ExpandDims:output:06basemodel/stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2"
 basemodel/stream_1_conv_1/conv1dб
(basemodel/stream_1_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€2*
(basemodel/stream_1_conv_1/conv1d/SqueezeЏ
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpх
!basemodel/stream_1_conv_1/BiasAddBiasAdd1basemodel/stream_1_conv_1/conv1d/Squeeze:output:08basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2#
!basemodel/stream_1_conv_1/BiasAdd≠
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€21
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimО
+basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_0_input_drop/Identity:output:08basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф2-
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
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1Я
 basemodel/stream_0_conv_1/conv1dConv2D4basemodel/stream_0_conv_1/conv1d/ExpandDims:output:06basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1dб
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€2*
(basemodel/stream_0_conv_1/conv1d/SqueezeЏ
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpх
!basemodel/stream_0_conv_1/BiasAddBiasAdd1basemodel/stream_0_conv_1/conv1d/Squeeze:output:08basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2#
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
-basemodel/batch_normalization_2/batchnorm/mul€
/basemodel/batch_normalization_2/batchnorm/mul_1Mul*basemodel/stream_2_conv_1/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@21
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
-basemodel/batch_normalization_2/batchnorm/subК
/basemodel/batch_normalization_2/batchnorm/add_1AddV23basemodel/batch_normalization_2/batchnorm/mul_1:z:01basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@21
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
-basemodel/batch_normalization_1/batchnorm/mul€
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_1_conv_1/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@21
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
-basemodel/batch_normalization_1/batchnorm/subК
/basemodel/batch_normalization_1/batchnorm/add_1AddV23basemodel/batch_normalization_1/batchnorm/mul_1:z:01basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@21
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
+basemodel/batch_normalization/batchnorm/mulщ
-basemodel/batch_normalization/batchnorm/mul_1Mul*basemodel/stream_0_conv_1/BiasAdd:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2/
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
+basemodel/batch_normalization/batchnorm/subВ
-basemodel/batch_normalization/batchnorm/add_1AddV21basemodel/batch_normalization/batchnorm/mul_1:z:0/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2/
-basemodel/batch_normalization/batchnorm/add_1Ѓ
basemodel/activation_2/TanhTanh3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
basemodel/activation_2/TanhЃ
basemodel/activation_1/TanhTanh3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
basemodel/activation_1/Tanh®
basemodel/activation/TanhTanh1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
basemodel/activation/TanhЬ
+basemodel/stream_2_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_2_maxpool_1/ExpandDims/dimт
'basemodel/stream_2_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation_2/Tanh:y:04basemodel/stream_2_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@2)
'basemodel/stream_2_maxpool_1/ExpandDimsч
$basemodel/stream_2_maxpool_1/MaxPoolMaxPool0basemodel/stream_2_maxpool_1/ExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_2_maxpool_1/MaxPool‘
$basemodel/stream_2_maxpool_1/SqueezeSqueeze-basemodel/stream_2_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2&
$basemodel/stream_2_maxpool_1/SqueezeЬ
+basemodel/stream_1_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_1_maxpool_1/ExpandDims/dimт
'basemodel/stream_1_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation_1/Tanh:y:04basemodel/stream_1_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@2)
'basemodel/stream_1_maxpool_1/ExpandDimsч
$basemodel/stream_1_maxpool_1/MaxPoolMaxPool0basemodel/stream_1_maxpool_1/ExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_1_maxpool_1/MaxPool‘
$basemodel/stream_1_maxpool_1/SqueezeSqueeze-basemodel/stream_1_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2&
$basemodel/stream_1_maxpool_1/SqueezeЬ
+basemodel/stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_0_maxpool_1/ExpandDims/dimр
'basemodel/stream_0_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation/Tanh:y:04basemodel/stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@2)
'basemodel/stream_0_maxpool_1/ExpandDimsч
$basemodel/stream_0_maxpool_1/MaxPoolMaxPool0basemodel/stream_0_maxpool_1/ExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_0_maxpool_1/MaxPool‘
$basemodel/stream_0_maxpool_1/SqueezeSqueeze-basemodel/stream_0_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2&
$basemodel/stream_0_maxpool_1/SqueezeЇ
"basemodel/stream_2_drop_1/IdentityIdentity-basemodel/stream_2_maxpool_1/Squeeze:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2$
"basemodel/stream_2_drop_1/IdentityЇ
"basemodel/stream_1_drop_1/IdentityIdentity-basemodel/stream_1_maxpool_1/Squeeze:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2$
"basemodel/stream_1_drop_1/IdentityЇ
"basemodel/stream_0_drop_1/IdentityIdentity-basemodel/stream_0_maxpool_1/Squeeze:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2$
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
:	ј@*
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
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Constш
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs≠
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1ў
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_1_conv_1/kernel/Regularizer/mulў
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addю
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square≠
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2а
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xд
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1Ў
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1Щ
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Constш
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs≠
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1ў
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_2_conv_1/kernel/Regularizer/mulў
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addю
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square≠
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2а
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xд
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1Ў
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const–
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ј@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2 
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
dense_1/kernel/Regularizer/add÷
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ј@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2#
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
NoOpNoOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp9^basemodel/batch_normalization/batchnorm/ReadVariableOp_19^basemodel/batch_normalization/batchnorm/ReadVariableOp_2;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_3/batchnorm/ReadVariableOp;^basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 2p
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
:€€€€€€€€€ф
 
_user_specified_nameinputs
љ
s
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_2258176

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
Л	
–
5__inference_batch_normalization_layer_call_fn_2262176

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
P__inference_batch_normalization_layer_call_and_return_conditional_losses_22576322
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
л
V
:__inference_global_average_pooling1d_layer_call_fn_2262829

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
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_22586532
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
С
n
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_2258375

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Ж+
й
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2259049

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
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
:€€€€€€€€€ф@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ф@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
Й
s
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_2262863

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
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
э!
ў
D__inference_dense_1_layer_call_and_return_conditional_losses_2262968

inputs1
matmul_readvariableop_resource:	ј@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ј@*
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
 dense_1/kernel/Regularizer/ConstЊ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ј@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2 
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
dense_1/kernel/Regularizer/addƒ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ј@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2#
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
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ј: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
ц
±
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2258214

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
®
з
)__inference_model_1_layer_call_fn_2260304
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

unknown_17:	ј@

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
	
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_22602002
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
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:€€€€€€€€€ф
%
_user_specified_nameleft_inputs
Ц
k
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_2262704

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
№
“
7__inference_batch_normalization_3_layer_call_fn_2262981

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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22582142
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
±Ё
®
F__inference_basemodel_layer_call_and_return_conditional_losses_2258793

inputs
inputs_1
inputs_2-
stream_2_conv_1_2258415:@%
stream_2_conv_1_2258417:@-
stream_1_conv_1_2258451:@%
stream_1_conv_1_2258453:@-
stream_0_conv_1_2258487:@%
stream_0_conv_1_2258489:@+
batch_normalization_2_2258512:@+
batch_normalization_2_2258514:@+
batch_normalization_2_2258516:@+
batch_normalization_2_2258518:@+
batch_normalization_1_2258541:@+
batch_normalization_1_2258543:@+
batch_normalization_1_2258545:@+
batch_normalization_1_2258547:@)
batch_normalization_2258570:@)
batch_normalization_2258572:@)
batch_normalization_2258574:@)
batch_normalization_2258576:@"
dense_1_2258712:	ј@
dense_1_2258714:@+
batch_normalization_3_2258717:@+
batch_normalization_3_2258719:@+
batch_normalization_3_2258721:@+
batch_normalization_3_2258723:@
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_conv_1/StatefulPartitionedCallҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_1_conv_1/StatefulPartitionedCallҐ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_2_conv_1/StatefulPartitionedCallҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpА
#stream_2_input_drop/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_22583682%
#stream_2_input_drop/PartitionedCallА
#stream_1_input_drop/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_22583752%
#stream_1_input_drop/PartitionedCallю
#stream_0_input_drop/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_22583822%
#stream_0_input_drop/PartitionedCallи
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_2_input_drop/PartitionedCall:output:0stream_2_conv_1_2258415stream_2_conv_1_2258417*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_22584142)
'stream_2_conv_1/StatefulPartitionedCallи
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_1_input_drop/PartitionedCall:output:0stream_1_conv_1_2258451stream_1_conv_1_2258453*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_22584502)
'stream_1_conv_1/StatefulPartitionedCallи
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_2258487stream_0_conv_1_2258489*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_22584862)
'stream_0_conv_1/StatefulPartitionedCallћ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_2258512batch_normalization_2_2258514batch_normalization_2_2258516batch_normalization_2_2258518*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22585112/
-batch_normalization_2/StatefulPartitionedCallћ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_2258541batch_normalization_1_2258543batch_normalization_1_2258545batch_normalization_1_2258547*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22585402/
-batch_normalization_1/StatefulPartitionedCallЊ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_2258570batch_normalization_2258572batch_normalization_2258574batch_normalization_2258576*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_22585692-
+batch_normalization/StatefulPartitionedCallЩ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_22585842
activation_2/PartitionedCallЩ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_22585912
activation_1/PartitionedCallС
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_22585982
activation/PartitionedCallЪ
"stream_2_maxpool_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_22586072$
"stream_2_maxpool_1/PartitionedCallЪ
"stream_1_maxpool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_22586162$
"stream_1_maxpool_1/PartitionedCallШ
"stream_0_maxpool_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_22586252$
"stream_0_maxpool_1/PartitionedCallЧ
stream_2_drop_1/PartitionedCallPartitionedCall+stream_2_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_22586322!
stream_2_drop_1/PartitionedCallЧ
stream_1_drop_1/PartitionedCallPartitionedCall+stream_1_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_22586392!
stream_1_drop_1/PartitionedCallЧ
stream_0_drop_1/PartitionedCallPartitionedCall+stream_0_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_22586462!
stream_0_drop_1/PartitionedCall™
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
GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_22586532*
(global_average_pooling1d/PartitionedCall∞
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
GPU2*0J 8В *`
f[RY
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_22586602,
*global_average_pooling1d_1/PartitionedCall∞
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
GPU2*0J 8В *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_22586672,
*global_average_pooling1d_2/PartitionedCallщ
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
GPU2*0J 8В *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_22586772
concatenate/PartitionedCallМ
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
GPU2*0J 8В *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_22586842!
dense_1_dropout/PartitionedCallЈ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_2258712dense_1_2258714*
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
D__inference_dense_1_layer_call_and_return_conditional_losses_22587112!
dense_1/StatefulPartitionedCallњ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_2258717batch_normalization_3_2258719batch_normalization_3_2258721batch_normalization_3_2258723*
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22582142/
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
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_22587302$
"dense_activation_1/PartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const 
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_2258487*"
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
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_1_2258487*"
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
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Const 
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_1_conv_1_2258451*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs≠
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1ў
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_1_conv_1/kernel/Regularizer/mulў
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/add–
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_2258451*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square≠
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2а
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xд
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1Ў
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1Щ
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Const 
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_2258415*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs≠
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1ў
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_2_conv_1/kernel/Regularizer/mulў
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/add–
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_2_conv_1_2258415*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square≠
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2а
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xд
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1Ў
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Constѓ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_2258712*
_output_shapes
:	ј@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2 
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
dense_1/kernel/Regularizer/addµ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_2258712*
_output_shapes
:	ј@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2#
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
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_1_conv_1/StatefulPartitionedCall6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:€€€€€€€€€ф:€€€€€€€€€ф:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:€€€€€€€€€ф
 
_user_specified_nameinputs:TP
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs:TP
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
э!
ў
D__inference_dense_1_layer_call_and_return_conditional_losses_2258711

inputs1
matmul_readvariableop_resource:	ј@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ј@*
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
 dense_1/kernel/Regularizer/ConstЊ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ј@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2 
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
dense_1/kernel/Regularizer/addƒ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ј@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2#
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
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ј: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
п
X
<__inference_global_average_pooling1d_1_layer_call_fn_2262851

inputs
identityЎ
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
GPU2*0J 8В *`
f[RY
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_22586602
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
э
j
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_2258684

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
Њ
В
+__inference_basemodel_layer_call_fn_2261357
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

unknown_17:	ј@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИҐStatefulPartitionedCallЉ
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
:€€€€€€€€€@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_22587932
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
_construction_contextkEagerRuntime*Л
_input_shapesz
x:€€€€€€€€€ф:€€€€€€€€€ф:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs/2
∞
з
)__inference_model_1_layer_call_fn_2260032
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

unknown_17:	ј@

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
D__inference_model_1_layer_call_and_return_conditional_losses_22599812
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
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:€€€€€€€€€ф
%
_user_specified_nameleft_inputs
Ц
k
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_2258074

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
j
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_2262807

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
Ќ*
л
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2258274

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
љ
s
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_2258152

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
ѓ
k
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_2262738

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimВ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@2

ExpandDims†
MaxPoolMaxPoolExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2	
MaxPool}
SqueezeSqueezeMaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2	
Squeezei
IdentityIdentitySqueeze:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф@:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
ъi
ї
#__inference__traced_restore_2263314
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
"assignvariableop_12_dense_1_kernel:	ј@.
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
valueХ	BТ	B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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

Identity_2Ѓ
AssignVariableOp_2AssignVariableOp)assignvariableop_2_stream_1_conv_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ђ
AssignVariableOp_3AssignVariableOp'assignvariableop_3_stream_1_conv_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ѓ
AssignVariableOp_4AssignVariableOp)assignvariableop_4_stream_2_conv_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ђ
AssignVariableOp_5AssignVariableOp'assignvariableop_5_stream_2_conv_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6±
AssignVariableOp_6AssignVariableOp,assignvariableop_6_batch_normalization_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7∞
AssignVariableOp_7AssignVariableOp+assignvariableop_7_batch_normalization_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8≥
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9≤
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
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
К
г
%__inference_signature_wrapper_2260645
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

unknown_17:	ј@

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
"__inference__wrapped_model_22575482
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
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:€€€€€€€€€ф
%
_user_specified_nameleft_inputs
С
n
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_2258382

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
ш
щ
__inference_loss_fn_0_2263077T
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
л
–
5__inference_batch_normalization_layer_call_fn_2262202

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
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_22590492
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ф@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
Є
±
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2257896

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
Ж+
й
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2262310

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
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
:€€€€€€€€€ф@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ф@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
є+
л
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2257794

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
З
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_2258653

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
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
ъ,
О
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_2258486

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
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф2
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
conv1d/ExpandDims_1Ј
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2	
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
(stream_0_conv_1/kernel/Regularizer/add_1p
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity€
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ф: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
г
M
1__inference_stream_2_drop_1_layer_call_fn_2262797

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_22586322
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
ґ
ѓ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2262222

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
ъ,
О
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_2262150

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф2
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
conv1d/ExpandDims_1Ј
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2	
BiasAddЩ
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Constё
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs≠
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1ў
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_2_conv_1/kernel/Regularizer/mulў
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addд
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square≠
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2а
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xд
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1Ў
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1p
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity€
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ф: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Я
V
:__inference_global_average_pooling1d_layer_call_fn_2262824

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
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_22581282
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
Ё
J
.__inference_activation_2_layer_call_fn_2262655

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_22585842
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф@:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
Њ
В
+__inference_basemodel_layer_call_fn_2258844
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

unknown_17:	ј@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИҐStatefulPartitionedCallЉ
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
:€€€€€€€€€@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_22587932
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
_construction_contextkEagerRuntime*Л
_input_shapesz
x:€€€€€€€€€ф:€€€€€€€€€ф:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs_2
є+
л
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2262416

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
Л
h
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_2262919

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
Э
Ґ
1__inference_stream_1_conv_1_layer_call_fn_2262066

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_22584502
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ф: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Н
ѓ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2262276

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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
:€€€€€€€€€ф@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ф@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
ц
k
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_2262819

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape‘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
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
dropout/GreaterEqual/y√
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
Џ
“
7__inference_batch_normalization_3_layer_call_fn_2262994

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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22582742
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
й
c
G__inference_activation_layer_call_and_return_conditional_losses_2258598

inputs
identityS
TanhTanhinputs*
T0*,
_output_shapes
:€€€€€€€€€ф@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф@:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
й
P
4__inference_stream_2_maxpool_1_layer_call_fn_2262722

inputs
identity’
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_22586072
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф@:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
ќ
n
5__inference_stream_1_input_drop_layer_call_fn_2261944

inputs
identityИҐStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_22592532
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
П
±
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2258511

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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
:€€€€€€€€€ф@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ф@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
н
–
5__inference_batch_normalization_layer_call_fn_2262189

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
 *,
_output_shapes
:€€€€€€€€€ф@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_22585692
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ф@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
≤Ћ
т
F__inference_basemodel_layer_call_and_return_conditional_losses_2261611
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
&dense_1_matmul_readvariableop_resource:	ј@5
'dense_1_biasadd_readvariableop_resource:@E
7batch_normalization_3_batchnorm_readvariableop_resource:@I
;batch_normalization_3_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_3_batchnorm_readvariableop_1_resource:@G
9batch_normalization_3_batchnorm_readvariableop_2_resource:@
identityИҐ,batch_normalization/batchnorm/ReadVariableOpҐ.batch_normalization/batchnorm/ReadVariableOp_1Ґ.batch_normalization/batchnorm/ReadVariableOp_2Ґ0batch_normalization/batchnorm/mul/ReadVariableOpҐ.batch_normalization_1/batchnorm/ReadVariableOpҐ0batch_normalization_1/batchnorm/ReadVariableOp_1Ґ0batch_normalization_1/batchnorm/ReadVariableOp_2Ґ2batch_normalization_1/batchnorm/mul/ReadVariableOpҐ.batch_normalization_2/batchnorm/ReadVariableOpҐ0batch_normalization_2/batchnorm/ReadVariableOp_1Ґ0batch_normalization_2/batchnorm/ReadVariableOp_2Ґ2batch_normalization_2/batchnorm/mul/ReadVariableOpҐ.batch_normalization_3/batchnorm/ReadVariableOpҐ0batch_normalization_3/batchnorm/ReadVariableOp_1Ґ0batch_normalization_3/batchnorm/ReadVariableOp_2Ґ2batch_normalization_3/batchnorm/mul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_0_conv_1/BiasAdd/ReadVariableOpҐ2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_1_conv_1/BiasAdd/ReadVariableOpҐ2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ&stream_2_conv_1/BiasAdd/ReadVariableOpҐ2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЙ
stream_2_input_drop/IdentityIdentityinputs_2*
T0*,
_output_shapes
:€€€€€€€€€ф2
stream_2_input_drop/IdentityЙ
stream_1_input_drop/IdentityIdentityinputs_1*
T0*,
_output_shapes
:€€€€€€€€€ф2
stream_1_input_drop/IdentityЙ
stream_0_input_drop/IdentityIdentityinputs_0*
T0*,
_output_shapes
:€€€€€€€€€ф2
stream_0_input_drop/IdentityЩ
%stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_2_conv_1/conv1d/ExpandDims/dimж
!stream_2_conv_1/conv1d/ExpandDims
ExpandDims%stream_2_input_drop/Identity:output:0.stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф2#
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
#stream_2_conv_1/conv1d/ExpandDims_1ч
stream_2_conv_1/conv1dConv2D*stream_2_conv_1/conv1d/ExpandDims:output:0,stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2
stream_2_conv_1/conv1d√
stream_2_conv_1/conv1d/SqueezeSqueezestream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€2 
stream_2_conv_1/conv1d/SqueezeЉ
&stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_2_conv_1/BiasAdd/ReadVariableOpЌ
stream_2_conv_1/BiasAddBiasAdd'stream_2_conv_1/conv1d/Squeeze:output:0.stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
stream_2_conv_1/BiasAddЩ
%stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_1_conv_1/conv1d/ExpandDims/dimж
!stream_1_conv_1/conv1d/ExpandDims
ExpandDims%stream_1_input_drop/Identity:output:0.stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф2#
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
#stream_1_conv_1/conv1d/ExpandDims_1ч
stream_1_conv_1/conv1dConv2D*stream_1_conv_1/conv1d/ExpandDims:output:0,stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2
stream_1_conv_1/conv1d√
stream_1_conv_1/conv1d/SqueezeSqueezestream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€2 
stream_1_conv_1/conv1d/SqueezeЉ
&stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_1_conv_1/BiasAdd/ReadVariableOpЌ
stream_1_conv_1/BiasAddBiasAdd'stream_1_conv_1/conv1d/Squeeze:output:0.stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
stream_1_conv_1/BiasAddЩ
%stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2'
%stream_0_conv_1/conv1d/ExpandDims/dimж
!stream_0_conv_1/conv1d/ExpandDims
ExpandDims%stream_0_input_drop/Identity:output:0.stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф2#
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
#stream_0_conv_1/conv1d/ExpandDims_1ч
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d√
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€2 
stream_0_conv_1/conv1d/SqueezeЉ
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOpЌ
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
#batch_normalization_2/batchnorm/mul„
%batch_normalization_2/batchnorm/mul_1Mul stream_2_conv_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2'
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
#batch_normalization_2/batchnorm/subв
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2'
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
#batch_normalization_1/batchnorm/mul„
%batch_normalization_1/batchnorm/mul_1Mul stream_1_conv_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2'
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
#batch_normalization_1/batchnorm/subв
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2'
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
!batch_normalization/batchnorm/mul—
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2%
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
!batch_normalization/batchnorm/subЏ
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2%
#batch_normalization/batchnorm/add_1Р
activation_2/TanhTanh)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
activation_2/TanhР
activation_1/TanhTanh)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
activation_1/TanhК
activation/TanhTanh'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
activation/TanhИ
!stream_2_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_2_maxpool_1/ExpandDims/dim 
stream_2_maxpool_1/ExpandDims
ExpandDimsactivation_2/Tanh:y:0*stream_2_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@2
stream_2_maxpool_1/ExpandDimsў
stream_2_maxpool_1/MaxPoolMaxPool&stream_2_maxpool_1/ExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2
stream_2_maxpool_1/MaxPoolґ
stream_2_maxpool_1/SqueezeSqueeze#stream_2_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2
stream_2_maxpool_1/SqueezeИ
!stream_1_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_1_maxpool_1/ExpandDims/dim 
stream_1_maxpool_1/ExpandDims
ExpandDimsactivation_1/Tanh:y:0*stream_1_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@2
stream_1_maxpool_1/ExpandDimsў
stream_1_maxpool_1/MaxPoolMaxPool&stream_1_maxpool_1/ExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2
stream_1_maxpool_1/MaxPoolґ
stream_1_maxpool_1/SqueezeSqueeze#stream_1_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2
stream_1_maxpool_1/SqueezeИ
!stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_1/ExpandDims/dim»
stream_0_maxpool_1/ExpandDims
ExpandDimsactivation/Tanh:y:0*stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@2
stream_0_maxpool_1/ExpandDimsў
stream_0_maxpool_1/MaxPoolMaxPool&stream_0_maxpool_1/ExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_1/MaxPoolґ
stream_0_maxpool_1/SqueezeSqueeze#stream_0_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2
stream_0_maxpool_1/SqueezeЬ
stream_2_drop_1/IdentityIdentity#stream_2_maxpool_1/Squeeze:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
stream_2_drop_1/IdentityЬ
stream_1_drop_1/IdentityIdentity#stream_1_maxpool_1/Squeeze:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
stream_1_drop_1/IdentityЬ
stream_0_drop_1/IdentityIdentity#stream_0_maxpool_1/Squeeze:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
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
:	ј@*
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
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Constо
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs≠
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1ў
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_1_conv_1/kernel/Regularizer/mulў
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addф
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square≠
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2а
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xд
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1Ў
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1Щ
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Constо
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs≠
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1ў
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_2_conv_1/kernel/Regularizer/mulў
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addф
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square≠
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2а
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xд
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1Ў
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const∆
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ј@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2 
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
dense_1/kernel/Regularizer/addћ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ј@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2#
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
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_1_conv_1/BiasAdd/ReadVariableOp3^stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_2_conv_1/BiasAdd/ReadVariableOp3^stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:€€€€€€€€€ф:€€€€€€€€€ф:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 2\
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
:€€€€€€€€€ф
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs/2
г
M
1__inference_stream_1_drop_1_layer_call_fn_2262770

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_22586392
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
Ќ*
л
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2263048

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
Л
k
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_2263057

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
ш
щ
__inference_loss_fn_2_2263117T
>stream_2_conv_1_kernel_regularizer_abs_readvariableop_resource:@
identityИҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЩ
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Constс
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_2_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs≠
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1ў
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_2_conv_1/kernel/Regularizer/mulў
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addч
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>stream_2_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square≠
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2а
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xд
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1Ў
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1v
IdentityIdentity,stream_2_conv_1/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityЅ
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
Н
j
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_2262780

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
л
e
I__inference_activation_2_layer_call_and_return_conditional_losses_2262660

inputs
identityS
TanhTanhinputs*
T0*,
_output_shapes
:€€€€€€€€€ф@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф@:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
ґ
ѓ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2257572

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
Є
±
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2257734

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
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_2262678

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
ъ
o
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_2261988

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape‘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф*
dtype0*
seedЈ*
seed2є2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/y√
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ф2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ф2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
цz
ш

D__inference_model_1_layer_call_and_return_conditional_losses_2260200

inputs'
basemodel_2260090:@
basemodel_2260092:@'
basemodel_2260094:@
basemodel_2260096:@'
basemodel_2260098:@
basemodel_2260100:@
basemodel_2260102:@
basemodel_2260104:@
basemodel_2260106:@
basemodel_2260108:@
basemodel_2260110:@
basemodel_2260112:@
basemodel_2260114:@
basemodel_2260116:@
basemodel_2260118:@
basemodel_2260120:@
basemodel_2260122:@
basemodel_2260124:@$
basemodel_2260126:	ј@
basemodel_2260128:@
basemodel_2260130:@
basemodel_2260132:@
basemodel_2260134:@
basemodel_2260136:@
identityИҐ!basemodel/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpч
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsbasemodel_2260090basemodel_2260092basemodel_2260094basemodel_2260096basemodel_2260098basemodel_2260100basemodel_2260102basemodel_2260104basemodel_2260106basemodel_2260108basemodel_2260110basemodel_2260112basemodel_2260114basemodel_2260116basemodel_2260118basemodel_2260120basemodel_2260122basemodel_2260124basemodel_2260126basemodel_2260128basemodel_2260130basemodel_2260132basemodel_2260134basemodel_2260136*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_22594782#
!basemodel/StatefulPartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constƒ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_2260098*"
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
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_2260098*"
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
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Constƒ
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_2260094*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs≠
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1ў
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_1_conv_1/kernel/Regularizer/mulў
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/add 
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_2260094*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square≠
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2а
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xд
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1Ў
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1Щ
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Constƒ
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_2260090*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs≠
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1ў
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_2_conv_1/kernel/Regularizer/mulў
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/add 
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_2260090*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square≠
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2а
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xд
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1Ў
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const±
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_2260126*
_output_shapes
:	ј@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2 
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
dense_1/kernel/Regularizer/addЈ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_2260126*
_output_shapes
:	ј@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2#
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
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 2F
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
:€€€€€€€€€ф
 
_user_specified_nameinputs
л
Q
5__inference_stream_1_input_drop_layer_call_fn_2261939

inputs
identity÷
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_22583752
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
с
“
7__inference_batch_normalization_2_layer_call_fn_2262509

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22585112
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ф@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
Ѓ
P
4__inference_stream_1_maxpool_1_layer_call_fn_2262691

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
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_22580742
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
ъ
o
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_2259230

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape‘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф*
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
dropout/GreaterEqual/y√
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ф2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ф2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Є
±
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2262382

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
П
±
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2262436

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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
:€€€€€€€€€ф@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ф@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
П
±
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2262596

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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
:€€€€€€€€€ф@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ф@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
ъ,
О
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_2258450

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpҐ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф2
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
conv1d/ExpandDims_1Ј
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2	
BiasAddЩ
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Constё
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs≠
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1ў
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_1_conv_1/kernel/Regularizer/mulў
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addд
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square≠
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2а
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xд
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1Ў
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1p
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity€
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ф: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
ъ
o
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_2259276

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape‘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф*
dtype0*
seedЈ*
seed2є2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/y√
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ф2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ф2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
ц
k
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_2258918

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape‘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
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
dropout/GreaterEqual/y√
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
ќ
n
5__inference_stream_2_input_drop_layer_call_fn_2261971

inputs
identityИҐStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_22592762
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Н
ѓ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2258569

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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
:€€€€€€€€€ф@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ф@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
ѓ
k
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_2258625

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimВ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@2

ExpandDims†
MaxPoolMaxPoolExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2	
MaxPool}
SqueezeSqueezeMaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2	
Squeezei
IdentityIdentitySqueeze:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф@:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
Ѓ
P
4__inference_stream_2_maxpool_1_layer_call_fn_2262717

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
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_22581022
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
Н
j
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_2258646

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
г
M
1__inference_stream_0_drop_1_layer_call_fn_2262743

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_22586462
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
л
e
I__inference_activation_1_layer_call_and_return_conditional_losses_2258591

inputs
identityS
TanhTanhinputs*
T0*,
_output_shapes
:€€€€€€€€€ф@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф@:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
ё
А
H__inference_concatenate_layer_call_and_return_conditional_losses_2258677

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
Н	
–
5__inference_batch_normalization_layer_call_fn_2262163

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
P__inference_batch_normalization_layer_call_and_return_conditional_losses_22575722
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
Н
j
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_2258632

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
£
X
<__inference_global_average_pooling1d_1_layer_call_fn_2262846

inputs
identityб
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
GPU2*0J 8В *`
f[RY
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_22581522
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
Ц
k
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_2262730

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
Л
k
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_2258730

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
п
X
<__inference_global_average_pooling1d_2_layer_call_fn_2262873

inputs
identityЎ
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
GPU2*0J 8В *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_22586672
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
Й
s
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_2262885

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
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
ў
H
,__inference_activation_layer_call_fn_2262635

inputs
identityЌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_22585982
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф@:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
°
в
)__inference_model_1_layer_call_fn_2260698

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

unknown_17:	ј@

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
D__inference_model_1_layer_call_and_return_conditional_losses_22599812
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
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
ц
k
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_2258941

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape‘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
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
dropout/GreaterEqual/y√
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
ѓ
k
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_2258607

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimВ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@2

ExpandDims†
MaxPoolMaxPoolExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2	
MaxPool}
SqueezeSqueezeMaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2	
Squeezei
IdentityIdentitySqueeze:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф@:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
С
n
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_2261922

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Ц
k
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_2258102

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
ш
щ
__inference_loss_fn_1_2263097T
>stream_1_conv_1_kernel_regularizer_abs_readvariableop_resource:@
identityИҐ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЩ
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Constс
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_1_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs≠
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1ў
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_1_conv_1/kernel/Regularizer/mulў
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addч
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>stream_1_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square≠
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2а
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xд
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1Ў
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1v
IdentityIdentity,stream_1_conv_1/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityЅ
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
”
M
1__inference_dense_1_dropout_layer_call_fn_2262910

inputs
identityќ
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
GPU2*0J 8В *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_22588722
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
∆
j
1__inference_stream_2_drop_1_layer_call_fn_2262802

inputs
identityИҐStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_22589642
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
ї
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_2258128

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
л
e
I__inference_activation_2_layer_call_and_return_conditional_losses_2258584

inputs
identityS
TanhTanhinputs*
T0*,
_output_shapes
:€€€€€€€€€ф@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф@:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
юz
ш

D__inference_model_1_layer_call_and_return_conditional_losses_2259981

inputs'
basemodel_2259871:@
basemodel_2259873:@'
basemodel_2259875:@
basemodel_2259877:@'
basemodel_2259879:@
basemodel_2259881:@
basemodel_2259883:@
basemodel_2259885:@
basemodel_2259887:@
basemodel_2259889:@
basemodel_2259891:@
basemodel_2259893:@
basemodel_2259895:@
basemodel_2259897:@
basemodel_2259899:@
basemodel_2259901:@
basemodel_2259903:@
basemodel_2259905:@$
basemodel_2259907:	ј@
basemodel_2259909:@
basemodel_2259911:@
basemodel_2259913:@
basemodel_2259915:@
basemodel_2259917:@
identityИҐ!basemodel/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp€
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsbasemodel_2259871basemodel_2259873basemodel_2259875basemodel_2259877basemodel_2259879basemodel_2259881basemodel_2259883basemodel_2259885basemodel_2259887basemodel_2259889basemodel_2259891basemodel_2259893basemodel_2259895basemodel_2259897basemodel_2259899basemodel_2259901basemodel_2259903basemodel_2259905basemodel_2259907basemodel_2259909basemodel_2259911basemodel_2259913basemodel_2259915basemodel_2259917*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_22587932#
!basemodel/StatefulPartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constƒ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_2259879*"
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
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_2259879*"
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
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Constƒ
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_2259875*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs≠
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1ў
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_1_conv_1/kernel/Regularizer/mulў
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/add 
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_2259875*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square≠
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2а
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xд
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1Ў
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1Щ
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Constƒ
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_2259871*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs≠
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1ў
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_2_conv_1/kernel/Regularizer/mulў
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/add 
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_2259871*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square≠
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2а
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xд
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1Ў
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const±
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_2259907*
_output_shapes
:	ј@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2 
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
dense_1/kernel/Regularizer/addЈ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_2259907*
_output_shapes
:	ј@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2#
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
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 2F
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
:€€€€€€€€€ф
 
_user_specified_nameinputs
°
ё
__inference_loss_fn_3_2263137I
6dense_1_kernel_regularizer_abs_readvariableop_resource:	ј@
identityИҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpЙ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const÷
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	ј@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2 
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
dense_1/kernel/Regularizer/add№
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	ј@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2#
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
™:
Б
 __inference__traced_save_2263232
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
valueХ	BТ	B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЇ
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

identity_1Identity_1:output:0*∆
_input_shapesі
±: :@:@:@:@:@:@:@:@:@:@:@:@:	ј@:@:@:@:@:@:@:@:@:@:@:@: 2(
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
:	ј@: 
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
ц
k
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_2258964

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape‘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
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
dropout/GreaterEqual/y√
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
П	
“
7__inference_batch_normalization_2_layer_call_fn_2262496

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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22579562
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
ъ,
О
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_2262042

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
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф2
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
conv1d/ExpandDims_1Ј
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2	
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
(stream_0_conv_1/kernel/Regularizer/add_1p
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity€
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ф: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
С
n
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_2261949

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
си
≤
F__inference_basemodel_layer_call_and_return_conditional_losses_2259864
inputs_0
inputs_1
inputs_2-
stream_2_conv_1_2259732:@%
stream_2_conv_1_2259734:@-
stream_1_conv_1_2259737:@%
stream_1_conv_1_2259739:@-
stream_0_conv_1_2259742:@%
stream_0_conv_1_2259744:@+
batch_normalization_2_2259747:@+
batch_normalization_2_2259749:@+
batch_normalization_2_2259751:@+
batch_normalization_2_2259753:@+
batch_normalization_1_2259756:@+
batch_normalization_1_2259758:@+
batch_normalization_1_2259760:@+
batch_normalization_1_2259762:@)
batch_normalization_2259765:@)
batch_normalization_2259767:@)
batch_normalization_2259769:@)
batch_normalization_2259771:@"
dense_1_2259788:	ј@
dense_1_2259790:@+
batch_normalization_3_2259793:@+
batch_normalization_3_2259795:@+
batch_normalization_3_2259797:@+
batch_normalization_3_2259799:@
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_conv_1/StatefulPartitionedCallҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_drop_1/StatefulPartitionedCallҐ+stream_0_input_drop/StatefulPartitionedCallҐ'stream_1_conv_1/StatefulPartitionedCallҐ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_1_drop_1/StatefulPartitionedCallҐ+stream_1_input_drop/StatefulPartitionedCallҐ'stream_2_conv_1/StatefulPartitionedCallҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_2_drop_1/StatefulPartitionedCallҐ+stream_2_input_drop/StatefulPartitionedCallШ
+stream_2_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_22592762-
+stream_2_input_drop/StatefulPartitionedCall∆
+stream_1_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_1,^stream_2_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_22592532-
+stream_1_input_drop/StatefulPartitionedCall∆
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_0,^stream_1_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_22592302-
+stream_0_input_drop/StatefulPartitionedCallр
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_2_input_drop/StatefulPartitionedCall:output:0stream_2_conv_1_2259732stream_2_conv_1_2259734*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_22584142)
'stream_2_conv_1/StatefulPartitionedCallр
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_1_input_drop/StatefulPartitionedCall:output:0stream_1_conv_1_2259737stream_1_conv_1_2259739*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_22584502)
'stream_1_conv_1/StatefulPartitionedCallр
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_2259742stream_0_conv_1_2259744*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_22584862)
'stream_0_conv_1/StatefulPartitionedCall 
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_2259747batch_normalization_2_2259749batch_normalization_2_2259751batch_normalization_2_2259753*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22591692/
-batch_normalization_2/StatefulPartitionedCall 
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_2259756batch_normalization_1_2259758batch_normalization_1_2259760batch_normalization_1_2259762*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22591092/
-batch_normalization_1/StatefulPartitionedCallЉ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_2259765batch_normalization_2259767batch_normalization_2259769batch_normalization_2259771*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_22590492-
+batch_normalization/StatefulPartitionedCallЩ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_22585842
activation_2/PartitionedCallЩ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_22585912
activation_1/PartitionedCallС
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_22585982
activation/PartitionedCallЪ
"stream_2_maxpool_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_22586072$
"stream_2_maxpool_1/PartitionedCallЪ
"stream_1_maxpool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_22586162$
"stream_1_maxpool_1/PartitionedCallШ
"stream_0_maxpool_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_22586252$
"stream_0_maxpool_1/PartitionedCallЁ
'stream_2_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_2_maxpool_1/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_22589642)
'stream_2_drop_1/StatefulPartitionedCallў
'stream_1_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_1_maxpool_1/PartitionedCall:output:0(^stream_2_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_22589412)
'stream_1_drop_1/StatefulPartitionedCallў
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_0_maxpool_1/PartitionedCall:output:0(^stream_1_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_22589182)
'stream_0_drop_1/StatefulPartitionedCall≤
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
GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_22586532*
(global_average_pooling1d/PartitionedCallЄ
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
GPU2*0J 8В *`
f[RY
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_22586602,
*global_average_pooling1d_1/PartitionedCallЄ
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
GPU2*0J 8В *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_22586672,
*global_average_pooling1d_2/PartitionedCallщ
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
GPU2*0J 8В *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_22586772
concatenate/PartitionedCallМ
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
GPU2*0J 8В *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_22588722!
dense_1_dropout/PartitionedCallЈ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_2259788dense_1_2259790*
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
D__inference_dense_1_layer_call_and_return_conditional_losses_22587112!
dense_1/StatefulPartitionedCallљ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_2259793batch_normalization_3_2259795batch_normalization_3_2259797batch_normalization_3_2259799*
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22582742/
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
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_22587302$
"dense_activation_1/PartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const 
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_2259742*"
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
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_1_2259742*"
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
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Const 
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_1_conv_1_2259737*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs≠
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1ў
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_1_conv_1/kernel/Regularizer/mulў
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/add–
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_2259737*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square≠
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2а
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xд
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1Ў
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1Щ
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Const 
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_2259732*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs≠
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1ў
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_2_conv_1/kernel/Regularizer/mulў
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/add–
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_2_conv_1_2259732*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square≠
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2а
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xд
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1Ў
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Constѓ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_2259788*
_output_shapes
:	ј@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2 
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
dense_1/kernel/Regularizer/addµ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_2259788*
_output_shapes
:	ј@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2#
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

Identityр
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall(^stream_1_conv_1/StatefulPartitionedCall6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_1_drop_1/StatefulPartitionedCall,^stream_1_input_drop/StatefulPartitionedCall(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_2_drop_1/StatefulPartitionedCall,^stream_2_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:€€€€€€€€€ф:€€€€€€€€€ф:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:€€€€€€€€€ф
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs_2
Н
j
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_2262753

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
ї
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_2262835

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
’
P
4__inference_dense_activation_1_layer_call_fn_2263053

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
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_22587302
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
ґ
В
+__inference_basemodel_layer_call_fn_2259584
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

unknown_17:	ј@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИҐStatefulPartitionedCallі
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
:€€€€€€€€€@*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_22594782
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
_construction_contextkEagerRuntime*Л
_input_shapesz
x:€€€€€€€€€ф:€€€€€€€€€ф:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs_2
ъ
o
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_2259253

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape‘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф*
dtype0*
seedЈ*
seed2Є2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/y√
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ф2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ф2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
ъ
o
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_2261934

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape‘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф*
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
dropout/GreaterEqual/y√
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ф2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ф2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Й
s
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_2258660

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
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
ѓ
k
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_2262712

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimВ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@2

ExpandDims†
MaxPoolMaxPoolExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2	
MaxPool}
SqueezeSqueezeMaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2	
Squeezei
IdentityIdentitySqueeze:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф@:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
И+
л
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2262470

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
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
:€€€€€€€€€ф@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ф@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
љ
s
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_2262857

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
л
Q
5__inference_stream_2_input_drop_layer_call_fn_2261966

inputs
identity÷
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_22583682
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
ш°
ў
"__inference__wrapped_model_2257548
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
8model_1_basemodel_dense_1_matmul_readvariableop_resource:	ј@G
9model_1_basemodel_dense_1_biasadd_readvariableop_resource:@W
Imodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_resource:@[
Mmodel_1_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:@Y
Kmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource:@Y
Kmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource:@
identityИҐ>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOpҐ@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1Ґ@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2ҐBmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpҐ@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpҐBmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1ҐBmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2ҐDmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpҐ@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOpҐBmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ҐBmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ҐDmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpҐ@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOpҐBmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1ҐBmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2ҐDmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpҐ0model_1/basemodel/dense_1/BiasAdd/ReadVariableOpҐ/model_1/basemodel/dense_1/MatMul/ReadVariableOpҐ8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpҐDmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ8model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpҐDmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ8model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpҐDmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp∞
.model_1/basemodel/stream_2_input_drop/IdentityIdentityleft_inputs*
T0*,
_output_shapes
:€€€€€€€€€ф20
.model_1/basemodel/stream_2_input_drop/Identity∞
.model_1/basemodel/stream_1_input_drop/IdentityIdentityleft_inputs*
T0*,
_output_shapes
:€€€€€€€€€ф20
.model_1/basemodel/stream_1_input_drop/Identity∞
.model_1/basemodel/stream_0_input_drop/IdentityIdentityleft_inputs*
T0*,
_output_shapes
:€€€€€€€€€ф20
.model_1/basemodel/stream_0_input_drop/Identityљ
7model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€29
7model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimЃ
3model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims
ExpandDims7model_1/basemodel/stream_2_input_drop/Identity:output:0@model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф25
3model_1/basemodel/stream_2_conv_1/conv1d/ExpandDimsЮ
Dmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpMmodel_1_basemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpЄ
9model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dimњ
5model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1
ExpandDimsLmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Bmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@27
5model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1њ
(model_1/basemodel/stream_2_conv_1/conv1dConv2D<model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims:output:0>model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2*
(model_1/basemodel/stream_2_conv_1/conv1dщ
0model_1/basemodel/stream_2_conv_1/conv1d/SqueezeSqueeze1model_1/basemodel/stream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€22
0model_1/basemodel/stream_2_conv_1/conv1d/Squeezeт
8model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOpAmodel_1_basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpХ
)model_1/basemodel/stream_2_conv_1/BiasAddBiasAdd9model_1/basemodel/stream_2_conv_1/conv1d/Squeeze:output:0@model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2+
)model_1/basemodel/stream_2_conv_1/BiasAddљ
7model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€29
7model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimЃ
3model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims
ExpandDims7model_1/basemodel/stream_1_input_drop/Identity:output:0@model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф25
3model_1/basemodel/stream_1_conv_1/conv1d/ExpandDimsЮ
Dmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpMmodel_1_basemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpЄ
9model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dimњ
5model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1
ExpandDimsLmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Bmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@27
5model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1њ
(model_1/basemodel/stream_1_conv_1/conv1dConv2D<model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims:output:0>model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2*
(model_1/basemodel/stream_1_conv_1/conv1dщ
0model_1/basemodel/stream_1_conv_1/conv1d/SqueezeSqueeze1model_1/basemodel/stream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€22
0model_1/basemodel/stream_1_conv_1/conv1d/Squeezeт
8model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOpAmodel_1_basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpХ
)model_1/basemodel/stream_1_conv_1/BiasAddBiasAdd9model_1/basemodel/stream_1_conv_1/conv1d/Squeeze:output:0@model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2+
)model_1/basemodel/stream_1_conv_1/BiasAddљ
7model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€29
7model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimЃ
3model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims7model_1/basemodel/stream_0_input_drop/Identity:output:0@model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф25
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
5model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1њ
(model_1/basemodel/stream_0_conv_1/conv1dConv2D<model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims:output:0>model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2*
(model_1/basemodel/stream_0_conv_1/conv1dщ
0model_1/basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze1model_1/basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€22
0model_1/basemodel/stream_0_conv_1/conv1d/Squeezeт
8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOpAmodel_1_basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpХ
)model_1/basemodel/stream_0_conv_1/BiasAddBiasAdd9model_1/basemodel/stream_0_conv_1/conv1d/Squeeze:output:0@model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2+
)model_1/basemodel/stream_0_conv_1/BiasAddК
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
5model_1/basemodel/batch_normalization_2/batchnorm/mulЯ
7model_1/basemodel/batch_normalization_2/batchnorm/mul_1Mul2model_1/basemodel/stream_2_conv_1/BiasAdd:output:09model_1/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@29
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
5model_1/basemodel/batch_normalization_2/batchnorm/sub™
7model_1/basemodel/batch_normalization_2/batchnorm/add_1AddV2;model_1/basemodel/batch_normalization_2/batchnorm/mul_1:z:09model_1/basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@29
7model_1/basemodel/batch_normalization_2/batchnorm/add_1К
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
5model_1/basemodel/batch_normalization_1/batchnorm/mulЯ
7model_1/basemodel/batch_normalization_1/batchnorm/mul_1Mul2model_1/basemodel/stream_1_conv_1/BiasAdd:output:09model_1/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@29
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
5model_1/basemodel/batch_normalization_1/batchnorm/sub™
7model_1/basemodel/batch_normalization_1/batchnorm/add_1AddV2;model_1/basemodel/batch_normalization_1/batchnorm/mul_1:z:09model_1/basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@29
7model_1/basemodel/batch_normalization_1/batchnorm/add_1Д
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
3model_1/basemodel/batch_normalization/batchnorm/mulЩ
5model_1/basemodel/batch_normalization/batchnorm/mul_1Mul2model_1/basemodel/stream_0_conv_1/BiasAdd:output:07model_1/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@27
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
3model_1/basemodel/batch_normalization/batchnorm/subҐ
5model_1/basemodel/batch_normalization/batchnorm/add_1AddV29model_1/basemodel/batch_normalization/batchnorm/mul_1:z:07model_1/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@27
5model_1/basemodel/batch_normalization/batchnorm/add_1∆
#model_1/basemodel/activation_2/TanhTanh;model_1/basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2%
#model_1/basemodel/activation_2/Tanh∆
#model_1/basemodel/activation_1/TanhTanh;model_1/basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2%
#model_1/basemodel/activation_1/Tanhј
!model_1/basemodel/activation/TanhTanh9model_1/basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2#
!model_1/basemodel/activation/Tanhђ
3model_1/basemodel/stream_2_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :25
3model_1/basemodel/stream_2_maxpool_1/ExpandDims/dimТ
/model_1/basemodel/stream_2_maxpool_1/ExpandDims
ExpandDims'model_1/basemodel/activation_2/Tanh:y:0<model_1/basemodel/stream_2_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@21
/model_1/basemodel/stream_2_maxpool_1/ExpandDimsП
,model_1/basemodel/stream_2_maxpool_1/MaxPoolMaxPool8model_1/basemodel/stream_2_maxpool_1/ExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2.
,model_1/basemodel/stream_2_maxpool_1/MaxPoolм
,model_1/basemodel/stream_2_maxpool_1/SqueezeSqueeze5model_1/basemodel/stream_2_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2.
,model_1/basemodel/stream_2_maxpool_1/Squeezeђ
3model_1/basemodel/stream_1_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :25
3model_1/basemodel/stream_1_maxpool_1/ExpandDims/dimТ
/model_1/basemodel/stream_1_maxpool_1/ExpandDims
ExpandDims'model_1/basemodel/activation_1/Tanh:y:0<model_1/basemodel/stream_1_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@21
/model_1/basemodel/stream_1_maxpool_1/ExpandDimsП
,model_1/basemodel/stream_1_maxpool_1/MaxPoolMaxPool8model_1/basemodel/stream_1_maxpool_1/ExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2.
,model_1/basemodel/stream_1_maxpool_1/MaxPoolм
,model_1/basemodel/stream_1_maxpool_1/SqueezeSqueeze5model_1/basemodel/stream_1_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2.
,model_1/basemodel/stream_1_maxpool_1/Squeezeђ
3model_1/basemodel/stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :25
3model_1/basemodel/stream_0_maxpool_1/ExpandDims/dimР
/model_1/basemodel/stream_0_maxpool_1/ExpandDims
ExpandDims%model_1/basemodel/activation/Tanh:y:0<model_1/basemodel/stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@21
/model_1/basemodel/stream_0_maxpool_1/ExpandDimsП
,model_1/basemodel/stream_0_maxpool_1/MaxPoolMaxPool8model_1/basemodel/stream_0_maxpool_1/ExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2.
,model_1/basemodel/stream_0_maxpool_1/MaxPoolм
,model_1/basemodel/stream_0_maxpool_1/SqueezeSqueeze5model_1/basemodel/stream_0_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2.
,model_1/basemodel/stream_0_maxpool_1/Squeeze“
*model_1/basemodel/stream_2_drop_1/IdentityIdentity5model_1/basemodel/stream_2_maxpool_1/Squeeze:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2,
*model_1/basemodel/stream_2_drop_1/Identity“
*model_1/basemodel/stream_1_drop_1/IdentityIdentity5model_1/basemodel/stream_1_maxpool_1/Squeeze:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2,
*model_1/basemodel/stream_1_drop_1/Identity“
*model_1/basemodel/stream_0_drop_1/IdentityIdentity5model_1/basemodel/stream_0_maxpool_1/Squeeze:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2,
*model_1/basemodel/stream_0_drop_1/Identity»
Amodel_1/basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Amodel_1/basemodel/global_average_pooling1d/Mean/reduction_indicesЭ
/model_1/basemodel/global_average_pooling1d/MeanMean3model_1/basemodel/stream_0_drop_1/Identity:output:0Jmodel_1/basemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@21
/model_1/basemodel/global_average_pooling1d/Meanћ
Cmodel_1/basemodel/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2E
Cmodel_1/basemodel/global_average_pooling1d_1/Mean/reduction_indices£
1model_1/basemodel/global_average_pooling1d_1/MeanMean3model_1/basemodel/stream_1_drop_1/Identity:output:0Lmodel_1/basemodel/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@23
1model_1/basemodel/global_average_pooling1d_1/Meanћ
Cmodel_1/basemodel/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2E
Cmodel_1/basemodel/global_average_pooling1d_2/Mean/reduction_indices£
1model_1/basemodel/global_average_pooling1d_2/MeanMean3model_1/basemodel/stream_2_drop_1/Identity:output:0Lmodel_1/basemodel/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€@23
1model_1/basemodel/global_average_pooling1d_2/MeanШ
)model_1/basemodel/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_1/basemodel/concatenate/concat/axisъ
$model_1/basemodel/concatenate/concatConcatV28model_1/basemodel/global_average_pooling1d/Mean:output:0:model_1/basemodel/global_average_pooling1d_1/Mean:output:0:model_1/basemodel/global_average_pooling1d_2/Mean:output:02model_1/basemodel/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€ј2&
$model_1/basemodel/concatenate/concat∆
*model_1/basemodel/dense_1_dropout/IdentityIdentity-model_1/basemodel/concatenate/concat:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2,
*model_1/basemodel/dense_1_dropout/Identity№
/model_1/basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp8model_1_basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ј@*
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
NoOpNoOp?^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOpA^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1A^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2C^model_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp1^model_1/basemodel/dense_1/BiasAdd/ReadVariableOp0^model_1/basemodel/dense_1/MatMul/ReadVariableOp9^model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp9^model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp9^model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 2А
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
:€€€€€€€€€ф
%
_user_specified_nameleft_inputs
Э
Ґ
1__inference_stream_2_conv_1_layer_call_fn_2262120

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_22584142
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ф: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
с
“
7__inference_batch_normalization_1_layer_call_fn_2262349

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22585402
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ф@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
∆
j
1__inference_stream_1_drop_1_layer_call_fn_2262775

inputs
identityИҐStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_22589412
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
С	
“
7__inference_batch_normalization_2_layer_call_fn_2262483

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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22578962
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
ъ
o
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_2261961

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape‘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф*
dtype0*
seedЈ*
seed2Є2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/y√
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ф2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ф2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
ц
k
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_2262765

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape‘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
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
dropout/GreaterEqual/y√
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
є+
л
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2257956

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
∆
j
1__inference_stream_0_drop_1_layer_call_fn_2262748

inputs
identityИҐStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_22589182
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
З
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_2262841

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
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
ц
k
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_2262792

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape‘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
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
dropout/GreaterEqual/y√
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
й
P
4__inference_stream_0_maxpool_1_layer_call_fn_2262670

inputs
identity’
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_22586252
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф@:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
ч
Ч
)__inference_dense_1_layer_call_fn_2262943

inputs
unknown:	ј@
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
D__inference_dense_1_layer_call_and_return_conditional_losses_22587112
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
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ј: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
И+
л
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2262630

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
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
:€€€€€€€€€ф@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ф@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
й
P
4__inference_stream_1_maxpool_1_layer_call_fn_2262696

inputs
identity’
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_22586162
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф@:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
љ
s
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_2262879

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
И+
л
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2259169

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
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
:€€€€€€€€€ф@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identityт
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ф@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
Ќ
g
-__inference_concatenate_layer_call_fn_2262892
inputs_0
inputs_1
inputs_2
identityв
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
GPU2*0J 8В *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_22586772
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
ъ,
О
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_2262096

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpҐ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф2
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
conv1d/ExpandDims_1Ј
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2	
BiasAddЩ
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Constё
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs≠
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1ў
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_1_conv_1/kernel/Regularizer/mulў
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addд
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square≠
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2а
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xд
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1Ў
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1p
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity€
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ф: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
ѓ
k
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_2262686

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimВ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@2

ExpandDims†
MaxPoolMaxPoolExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2	
MaxPool}
SqueezeSqueezeMaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2	
Squeezei
IdentityIdentitySqueeze:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф@:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
ќ
n
5__inference_stream_0_input_drop_layer_call_fn_2261917

inputs
identityИҐStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_22592302
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Ј+
й
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2262256

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
П
±
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2258540

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
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
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
:€€€€€€€€€ф@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity¬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ф@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
Ѓ
P
4__inference_stream_0_maxpool_1_layer_call_fn_2262665

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
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_22580462
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
Н
j
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_2258639

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
≤Ћ
Ц"
D__inference_model_1_layer_call_and_return_conditional_losses_2261242

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
0basemodel_dense_1_matmul_readvariableop_resource:	ј@?
1basemodel_dense_1_biasadd_readvariableop_resource:@U
Gbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource:@W
Ibasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource:@S
Ebasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:@O
Abasemodel_batch_normalization_3_batchnorm_readvariableop_resource:@
identityИҐ-basemodel/batch_normalization/AssignMovingAvgҐ<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOpҐ/basemodel/batch_normalization/AssignMovingAvg_1Ґ>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOpҐ6basemodel/batch_normalization/batchnorm/ReadVariableOpҐ:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpҐ/basemodel/batch_normalization_1/AssignMovingAvgҐ>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOpҐ1basemodel/batch_normalization_1/AssignMovingAvg_1Ґ@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpҐ8basemodel/batch_normalization_1/batchnorm/ReadVariableOpҐ<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpҐ/basemodel/batch_normalization_2/AssignMovingAvgҐ>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOpҐ1basemodel/batch_normalization_2/AssignMovingAvg_1Ґ@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpҐ8basemodel/batch_normalization_2/batchnorm/ReadVariableOpҐ<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpҐ/basemodel/batch_normalization_3/AssignMovingAvgҐ>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOpҐ1basemodel/batch_normalization_3/AssignMovingAvg_1Ґ@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpҐ8basemodel/batch_normalization_3/batchnorm/ReadVariableOpҐ<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpҐ(basemodel/dense_1/BiasAdd/ReadVariableOpҐ'basemodel/dense_1/MatMul/ReadVariableOpҐ0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpҐ<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpҐ<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpҐ<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
+basemodel/stream_2_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2-
+basemodel/stream_2_input_drop/dropout/Const“
)basemodel/stream_2_input_drop/dropout/MulMulinputs4basemodel/stream_2_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2+
)basemodel/stream_2_input_drop/dropout/MulР
+basemodel/stream_2_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2-
+basemodel/stream_2_input_drop/dropout/ShapeЃ
Bbasemodel/stream_2_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_2_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф*
dtype0*
seedЈ*
seed2є2D
Bbasemodel/stream_2_input_drop/dropout/random_uniform/RandomUniform±
4basemodel/stream_2_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>26
4basemodel/stream_2_input_drop/dropout/GreaterEqual/yї
2basemodel/stream_2_input_drop/dropout/GreaterEqualGreaterEqualKbasemodel/stream_2_input_drop/dropout/random_uniform/RandomUniform:output:0=basemodel/stream_2_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф24
2basemodel/stream_2_input_drop/dropout/GreaterEqualё
*basemodel/stream_2_input_drop/dropout/CastCast6basemodel/stream_2_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ф2,
*basemodel/stream_2_input_drop/dropout/Castч
+basemodel/stream_2_input_drop/dropout/Mul_1Mul-basemodel/stream_2_input_drop/dropout/Mul:z:0.basemodel/stream_2_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ф2-
+basemodel/stream_2_input_drop/dropout/Mul_1Я
+basemodel/stream_1_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2-
+basemodel/stream_1_input_drop/dropout/Const“
)basemodel/stream_1_input_drop/dropout/MulMulinputs4basemodel/stream_1_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2+
)basemodel/stream_1_input_drop/dropout/MulР
+basemodel/stream_1_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2-
+basemodel/stream_1_input_drop/dropout/ShapeЃ
Bbasemodel/stream_1_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_1_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф*
dtype0*
seedЈ*
seed2Є2D
Bbasemodel/stream_1_input_drop/dropout/random_uniform/RandomUniform±
4basemodel/stream_1_input_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>26
4basemodel/stream_1_input_drop/dropout/GreaterEqual/yї
2basemodel/stream_1_input_drop/dropout/GreaterEqualGreaterEqualKbasemodel/stream_1_input_drop/dropout/random_uniform/RandomUniform:output:0=basemodel/stream_1_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф24
2basemodel/stream_1_input_drop/dropout/GreaterEqualё
*basemodel/stream_1_input_drop/dropout/CastCast6basemodel/stream_1_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ф2,
*basemodel/stream_1_input_drop/dropout/Castч
+basemodel/stream_1_input_drop/dropout/Mul_1Mul-basemodel/stream_1_input_drop/dropout/Mul:z:0.basemodel/stream_1_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ф2-
+basemodel/stream_1_input_drop/dropout/Mul_1Я
+basemodel/stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2-
+basemodel/stream_0_input_drop/dropout/Const“
)basemodel/stream_0_input_drop/dropout/MulMulinputs4basemodel/stream_0_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2+
)basemodel/stream_0_input_drop/dropout/MulР
+basemodel/stream_0_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2-
+basemodel/stream_0_input_drop/dropout/ShapeЃ
Bbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф*
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
4basemodel/stream_0_input_drop/dropout/GreaterEqual/yї
2basemodel/stream_0_input_drop/dropout/GreaterEqualGreaterEqualKbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniform:output:0=basemodel/stream_0_input_drop/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф24
2basemodel/stream_0_input_drop/dropout/GreaterEqualё
*basemodel/stream_0_input_drop/dropout/CastCast6basemodel/stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ф2,
*basemodel/stream_0_input_drop/dropout/Castч
+basemodel/stream_0_input_drop/dropout/Mul_1Mul-basemodel/stream_0_input_drop/dropout/Mul:z:0.basemodel/stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ф2-
+basemodel/stream_0_input_drop/dropout/Mul_1≠
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€21
/basemodel/stream_2_conv_1/conv1d/ExpandDims/dimО
+basemodel/stream_2_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_2_input_drop/dropout/Mul_1:z:08basemodel/stream_2_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф2-
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
-basemodel/stream_2_conv_1/conv1d/ExpandDims_1Я
 basemodel/stream_2_conv_1/conv1dConv2D4basemodel/stream_2_conv_1/conv1d/ExpandDims:output:06basemodel/stream_2_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2"
 basemodel/stream_2_conv_1/conv1dб
(basemodel/stream_2_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€2*
(basemodel/stream_2_conv_1/conv1d/SqueezeЏ
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpх
!basemodel/stream_2_conv_1/BiasAddBiasAdd1basemodel/stream_2_conv_1/conv1d/Squeeze:output:08basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2#
!basemodel/stream_2_conv_1/BiasAdd≠
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€21
/basemodel/stream_1_conv_1/conv1d/ExpandDims/dimО
+basemodel/stream_1_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_1_input_drop/dropout/Mul_1:z:08basemodel/stream_1_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф2-
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
-basemodel/stream_1_conv_1/conv1d/ExpandDims_1Я
 basemodel/stream_1_conv_1/conv1dConv2D4basemodel/stream_1_conv_1/conv1d/ExpandDims:output:06basemodel/stream_1_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2"
 basemodel/stream_1_conv_1/conv1dб
(basemodel/stream_1_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€2*
(basemodel/stream_1_conv_1/conv1d/SqueezeЏ
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpх
!basemodel/stream_1_conv_1/BiasAddBiasAdd1basemodel/stream_1_conv_1/conv1d/Squeeze:output:08basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2#
!basemodel/stream_1_conv_1/BiasAdd≠
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€21
/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimО
+basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims/basemodel/stream_0_input_drop/dropout/Mul_1:z:08basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф2-
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
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1Я
 basemodel/stream_0_conv_1/conv1dConv2D4basemodel/stream_0_conv_1/conv1d/ExpandDims:output:06basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1dб
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€2*
(basemodel/stream_0_conv_1/conv1d/SqueezeЏ
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpх
!basemodel/stream_0_conv_1/BiasAddBiasAdd1basemodel/stream_0_conv_1/conv1d/Squeeze:output:08basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2#
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
4basemodel/batch_normalization_2/moments/StopGradient≠
9basemodel/batch_normalization_2/moments/SquaredDifferenceSquaredDifference*basemodel/stream_2_conv_1/BiasAdd:output:0=basemodel/batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2;
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
-basemodel/batch_normalization_2/batchnorm/mul€
/basemodel/batch_normalization_2/batchnorm/mul_1Mul*basemodel/stream_2_conv_1/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@21
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
-basemodel/batch_normalization_2/batchnorm/subК
/basemodel/batch_normalization_2/batchnorm/add_1AddV23basemodel/batch_normalization_2/batchnorm/mul_1:z:01basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@21
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
4basemodel/batch_normalization_1/moments/StopGradient≠
9basemodel/batch_normalization_1/moments/SquaredDifferenceSquaredDifference*basemodel/stream_1_conv_1/BiasAdd:output:0=basemodel/batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2;
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
-basemodel/batch_normalization_1/batchnorm/mul€
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_1_conv_1/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@21
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
-basemodel/batch_normalization_1/batchnorm/subК
/basemodel/batch_normalization_1/batchnorm/add_1AddV23basemodel/batch_normalization_1/batchnorm/mul_1:z:01basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@21
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
2basemodel/batch_normalization/moments/StopGradientІ
7basemodel/batch_normalization/moments/SquaredDifferenceSquaredDifference*basemodel/stream_0_conv_1/BiasAdd:output:0;basemodel/batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@29
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
+basemodel/batch_normalization/batchnorm/mulщ
-basemodel/batch_normalization/batchnorm/mul_1Mul*basemodel/stream_0_conv_1/BiasAdd:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2/
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
+basemodel/batch_normalization/batchnorm/subВ
-basemodel/batch_normalization/batchnorm/add_1AddV21basemodel/batch_normalization/batchnorm/mul_1:z:0/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2/
-basemodel/batch_normalization/batchnorm/add_1Ѓ
basemodel/activation_2/TanhTanh3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
basemodel/activation_2/TanhЃ
basemodel/activation_1/TanhTanh3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
basemodel/activation_1/Tanh®
basemodel/activation/TanhTanh1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2
basemodel/activation/TanhЬ
+basemodel/stream_2_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_2_maxpool_1/ExpandDims/dimт
'basemodel/stream_2_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation_2/Tanh:y:04basemodel/stream_2_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@2)
'basemodel/stream_2_maxpool_1/ExpandDimsч
$basemodel/stream_2_maxpool_1/MaxPoolMaxPool0basemodel/stream_2_maxpool_1/ExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_2_maxpool_1/MaxPool‘
$basemodel/stream_2_maxpool_1/SqueezeSqueeze-basemodel/stream_2_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2&
$basemodel/stream_2_maxpool_1/SqueezeЬ
+basemodel/stream_1_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_1_maxpool_1/ExpandDims/dimт
'basemodel/stream_1_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation_1/Tanh:y:04basemodel/stream_1_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@2)
'basemodel/stream_1_maxpool_1/ExpandDimsч
$basemodel/stream_1_maxpool_1/MaxPoolMaxPool0basemodel/stream_1_maxpool_1/ExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_1_maxpool_1/MaxPool‘
$basemodel/stream_1_maxpool_1/SqueezeSqueeze-basemodel/stream_1_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2&
$basemodel/stream_1_maxpool_1/SqueezeЬ
+basemodel/stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_0_maxpool_1/ExpandDims/dimр
'basemodel/stream_0_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation/Tanh:y:04basemodel/stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@2)
'basemodel/stream_0_maxpool_1/ExpandDimsч
$basemodel/stream_0_maxpool_1/MaxPoolMaxPool0basemodel/stream_0_maxpool_1/ExpandDims:output:0*0
_output_shapes
:€€€€€€€€€ъ@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_0_maxpool_1/MaxPool‘
$basemodel/stream_0_maxpool_1/SqueezeSqueeze-basemodel/stream_0_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
squeeze_dims
2&
$basemodel/stream_0_maxpool_1/SqueezeЧ
'basemodel/stream_2_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2)
'basemodel/stream_2_drop_1/dropout/Constн
%basemodel/stream_2_drop_1/dropout/MulMul-basemodel/stream_2_maxpool_1/Squeeze:output:00basemodel/stream_2_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2'
%basemodel/stream_2_drop_1/dropout/Mulѓ
'basemodel/stream_2_drop_1/dropout/ShapeShape-basemodel/stream_2_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2)
'basemodel/stream_2_drop_1/dropout/ShapeҐ
>basemodel/stream_2_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_2_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
dtype0*
seedЈ*
seed2Ї2@
>basemodel/stream_2_drop_1/dropout/random_uniform/RandomUniform©
0basemodel/stream_2_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>22
0basemodel/stream_2_drop_1/dropout/GreaterEqual/yЂ
.basemodel/stream_2_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_2_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_2_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@20
.basemodel/stream_2_drop_1/dropout/GreaterEqual“
&basemodel/stream_2_drop_1/dropout/CastCast2basemodel/stream_2_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ъ@2(
&basemodel/stream_2_drop_1/dropout/Castз
'basemodel/stream_2_drop_1/dropout/Mul_1Mul)basemodel/stream_2_drop_1/dropout/Mul:z:0*basemodel/stream_2_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2)
'basemodel/stream_2_drop_1/dropout/Mul_1Ч
'basemodel/stream_1_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2)
'basemodel/stream_1_drop_1/dropout/Constн
%basemodel/stream_1_drop_1/dropout/MulMul-basemodel/stream_1_maxpool_1/Squeeze:output:00basemodel/stream_1_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2'
%basemodel/stream_1_drop_1/dropout/Mulѓ
'basemodel/stream_1_drop_1/dropout/ShapeShape-basemodel/stream_1_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2)
'basemodel/stream_1_drop_1/dropout/ShapeҐ
>basemodel/stream_1_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_1_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
dtype0*
seedЈ*
seed2є2@
>basemodel/stream_1_drop_1/dropout/random_uniform/RandomUniform©
0basemodel/stream_1_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>22
0basemodel/stream_1_drop_1/dropout/GreaterEqual/yЂ
.basemodel/stream_1_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_1_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_1_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@20
.basemodel/stream_1_drop_1/dropout/GreaterEqual“
&basemodel/stream_1_drop_1/dropout/CastCast2basemodel/stream_1_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ъ@2(
&basemodel/stream_1_drop_1/dropout/Castз
'basemodel/stream_1_drop_1/dropout/Mul_1Mul)basemodel/stream_1_drop_1/dropout/Mul:z:0*basemodel/stream_1_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2)
'basemodel/stream_1_drop_1/dropout/Mul_1Ч
'basemodel/stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ђ™™?2)
'basemodel/stream_0_drop_1/dropout/Constн
%basemodel/stream_0_drop_1/dropout/MulMul-basemodel/stream_0_maxpool_1/Squeeze:output:00basemodel/stream_0_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2'
%basemodel/stream_0_drop_1/dropout/Mulѓ
'basemodel/stream_0_drop_1/dropout/ShapeShape-basemodel/stream_0_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_1/dropout/ShapeҐ
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@*
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
0basemodel/stream_0_drop_1/dropout/GreaterEqual/yЂ
.basemodel/stream_0_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_0_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@20
.basemodel/stream_0_drop_1/dropout/GreaterEqual“
&basemodel/stream_0_drop_1/dropout/CastCast2basemodel/stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€ъ@2(
&basemodel/stream_0_drop_1/dropout/Castз
'basemodel/stream_0_drop_1/dropout/Mul_1Mul)basemodel/stream_0_drop_1/dropout/Mul:z:0*basemodel/stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€ъ@2)
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
:	ј@*
dtype02)
'basemodel/dense_1/MatMul/ReadVariableOp»
basemodel/dense_1/MatMulMatMul%basemodel/concatenate/concat:output:0/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
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
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Constш
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs≠
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1ў
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_1_conv_1/kernel/Regularizer/mulў
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addю
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square≠
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2а
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xд
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1Ў
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1Щ
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Constш
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs≠
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1ў
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_2_conv_1/kernel/Regularizer/mulў
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addю
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square≠
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2а
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xд
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1Ў
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const–
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ј@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2 
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
dense_1/kernel/Regularizer/add÷
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ј@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2#
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
NoOpNoOp.^basemodel/batch_normalization/AssignMovingAvg=^basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_1?^basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_1/AssignMovingAvg?^basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_1A^basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_2/AssignMovingAvg?^basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_1A^basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_3/AssignMovingAvg?^basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_3/AssignMovingAvg_1A^basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_3/batchnorm/ReadVariableOp=^basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 2^
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
:€€€€€€€€€ф
 
_user_specified_nameinputs
їЁ
™
F__inference_basemodel_layer_call_and_return_conditional_losses_2259724
inputs_0
inputs_1
inputs_2-
stream_2_conv_1_2259592:@%
stream_2_conv_1_2259594:@-
stream_1_conv_1_2259597:@%
stream_1_conv_1_2259599:@-
stream_0_conv_1_2259602:@%
stream_0_conv_1_2259604:@+
batch_normalization_2_2259607:@+
batch_normalization_2_2259609:@+
batch_normalization_2_2259611:@+
batch_normalization_2_2259613:@+
batch_normalization_1_2259616:@+
batch_normalization_1_2259618:@+
batch_normalization_1_2259620:@+
batch_normalization_1_2259622:@)
batch_normalization_2259625:@)
batch_normalization_2259627:@)
batch_normalization_2259629:@)
batch_normalization_2259631:@"
dense_1_2259648:	ј@
dense_1_2259650:@+
batch_normalization_3_2259653:@+
batch_normalization_3_2259655:@+
batch_normalization_3_2259657:@+
batch_normalization_3_2259659:@
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_conv_1/StatefulPartitionedCallҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_1_conv_1/StatefulPartitionedCallҐ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_2_conv_1/StatefulPartitionedCallҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpА
#stream_2_input_drop/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_22583682%
#stream_2_input_drop/PartitionedCallА
#stream_1_input_drop/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_22583752%
#stream_1_input_drop/PartitionedCallА
#stream_0_input_drop/PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_22583822%
#stream_0_input_drop/PartitionedCallи
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_2_input_drop/PartitionedCall:output:0stream_2_conv_1_2259592stream_2_conv_1_2259594*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_22584142)
'stream_2_conv_1/StatefulPartitionedCallи
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_1_input_drop/PartitionedCall:output:0stream_1_conv_1_2259597stream_1_conv_1_2259599*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_22584502)
'stream_1_conv_1/StatefulPartitionedCallи
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_2259602stream_0_conv_1_2259604*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_22584862)
'stream_0_conv_1/StatefulPartitionedCallћ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_2259607batch_normalization_2_2259609batch_normalization_2_2259611batch_normalization_2_2259613*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22585112/
-batch_normalization_2/StatefulPartitionedCallћ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_2259616batch_normalization_1_2259618batch_normalization_1_2259620batch_normalization_1_2259622*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22585402/
-batch_normalization_1/StatefulPartitionedCallЊ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_2259625batch_normalization_2259627batch_normalization_2259629batch_normalization_2259631*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_22585692-
+batch_normalization/StatefulPartitionedCallЩ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_22585842
activation_2/PartitionedCallЩ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_22585912
activation_1/PartitionedCallС
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_22585982
activation/PartitionedCallЪ
"stream_2_maxpool_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_22586072$
"stream_2_maxpool_1/PartitionedCallЪ
"stream_1_maxpool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_22586162$
"stream_1_maxpool_1/PartitionedCallШ
"stream_0_maxpool_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_22586252$
"stream_0_maxpool_1/PartitionedCallЧ
stream_2_drop_1/PartitionedCallPartitionedCall+stream_2_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_22586322!
stream_2_drop_1/PartitionedCallЧ
stream_1_drop_1/PartitionedCallPartitionedCall+stream_1_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_22586392!
stream_1_drop_1/PartitionedCallЧ
stream_0_drop_1/PartitionedCallPartitionedCall+stream_0_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_22586462!
stream_0_drop_1/PartitionedCall™
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
GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_22586532*
(global_average_pooling1d/PartitionedCall∞
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
GPU2*0J 8В *`
f[RY
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_22586602,
*global_average_pooling1d_1/PartitionedCall∞
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
GPU2*0J 8В *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_22586672,
*global_average_pooling1d_2/PartitionedCallщ
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
GPU2*0J 8В *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_22586772
concatenate/PartitionedCallМ
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
GPU2*0J 8В *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_22586842!
dense_1_dropout/PartitionedCallЈ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_2259648dense_1_2259650*
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
D__inference_dense_1_layer_call_and_return_conditional_losses_22587112!
dense_1/StatefulPartitionedCallњ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_2259653batch_normalization_3_2259655batch_normalization_3_2259657batch_normalization_3_2259659*
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22582142/
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
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_22587302$
"dense_activation_1/PartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const 
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_2259602*"
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
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_1_2259602*"
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
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Const 
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_1_conv_1_2259597*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs≠
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1ў
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_1_conv_1/kernel/Regularizer/mulў
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/add–
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_2259597*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square≠
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2а
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xд
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1Ў
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1Щ
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Const 
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_2259592*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs≠
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1ў
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_2_conv_1/kernel/Regularizer/mulў
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/add–
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_2_conv_1_2259592*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square≠
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2а
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xд
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1Ў
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Constѓ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_2259648*
_output_shapes
:	ј@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2 
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
dense_1/kernel/Regularizer/addµ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_2259648*
_output_shapes
:	ј@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2#
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
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_1_conv_1/StatefulPartitionedCall6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:€€€€€€€€€ф:€€€€€€€€€ф:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:€€€€€€€€€ф
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs_2
й
c
G__inference_activation_layer_call_and_return_conditional_losses_2262640

inputs
identityS
TanhTanhinputs*
T0*,
_output_shapes
:€€€€€€€€€ф@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф@:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
П	
“
7__inference_batch_normalization_1_layer_call_fn_2262336

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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22577942
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
ъ,
О
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_2258414

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф2
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
conv1d/ExpandDims_1Ј
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ф@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф@*
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ф@2	
BiasAddЩ
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Constё
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs≠
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1ў
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_2_conv_1/kernel/Regularizer/mulў
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addд
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square≠
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2а
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xд
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1Ў
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1p
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identity€
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ф: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
ґ
В
+__inference_basemodel_layer_call_fn_2261412
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

unknown_17:	ј@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИҐStatefulPartitionedCallі
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
:€€€€€€€€€@*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_22594782
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
_construction_contextkEagerRuntime*Л
_input_shapesz
x:€€€€€€€€€ф:€€€€€€€€€ф:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:€€€€€€€€€ф
"
_user_specified_name
inputs/2
зи
∞
F__inference_basemodel_layer_call_and_return_conditional_losses_2259478

inputs
inputs_1
inputs_2-
stream_2_conv_1_2259346:@%
stream_2_conv_1_2259348:@-
stream_1_conv_1_2259351:@%
stream_1_conv_1_2259353:@-
stream_0_conv_1_2259356:@%
stream_0_conv_1_2259358:@+
batch_normalization_2_2259361:@+
batch_normalization_2_2259363:@+
batch_normalization_2_2259365:@+
batch_normalization_2_2259367:@+
batch_normalization_1_2259370:@+
batch_normalization_1_2259372:@+
batch_normalization_1_2259374:@+
batch_normalization_1_2259376:@)
batch_normalization_2259379:@)
batch_normalization_2259381:@)
batch_normalization_2259383:@)
batch_normalization_2259385:@"
dense_1_2259402:	ј@
dense_1_2259404:@+
batch_normalization_3_2259407:@+
batch_normalization_3_2259409:@+
batch_normalization_3_2259411:@+
batch_normalization_3_2259413:@
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_conv_1/StatefulPartitionedCallҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_0_drop_1/StatefulPartitionedCallҐ+stream_0_input_drop/StatefulPartitionedCallҐ'stream_1_conv_1/StatefulPartitionedCallҐ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_1_drop_1/StatefulPartitionedCallҐ+stream_1_input_drop/StatefulPartitionedCallҐ'stream_2_conv_1/StatefulPartitionedCallҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ'stream_2_drop_1/StatefulPartitionedCallҐ+stream_2_input_drop/StatefulPartitionedCallШ
+stream_2_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_22592762-
+stream_2_input_drop/StatefulPartitionedCall∆
+stream_1_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_1,^stream_2_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_22592532-
+stream_1_input_drop/StatefulPartitionedCallƒ
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs,^stream_1_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_22592302-
+stream_0_input_drop/StatefulPartitionedCallр
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_2_input_drop/StatefulPartitionedCall:output:0stream_2_conv_1_2259346stream_2_conv_1_2259348*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_22584142)
'stream_2_conv_1/StatefulPartitionedCallр
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_1_input_drop/StatefulPartitionedCall:output:0stream_1_conv_1_2259351stream_1_conv_1_2259353*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_22584502)
'stream_1_conv_1/StatefulPartitionedCallр
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_2259356stream_0_conv_1_2259358*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_22584862)
'stream_0_conv_1/StatefulPartitionedCall 
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_2259361batch_normalization_2_2259363batch_normalization_2_2259365batch_normalization_2_2259367*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22591692/
-batch_normalization_2/StatefulPartitionedCall 
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_2259370batch_normalization_1_2259372batch_normalization_1_2259374batch_normalization_1_2259376*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22591092/
-batch_normalization_1/StatefulPartitionedCallЉ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_2259379batch_normalization_2259381batch_normalization_2259383batch_normalization_2259385*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_22590492-
+batch_normalization/StatefulPartitionedCallЩ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_22585842
activation_2/PartitionedCallЩ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_22585912
activation_1/PartitionedCallС
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_22585982
activation/PartitionedCallЪ
"stream_2_maxpool_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_22586072$
"stream_2_maxpool_1/PartitionedCallЪ
"stream_1_maxpool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_22586162$
"stream_1_maxpool_1/PartitionedCallШ
"stream_0_maxpool_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_22586252$
"stream_0_maxpool_1/PartitionedCallЁ
'stream_2_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_2_maxpool_1/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_22589642)
'stream_2_drop_1/StatefulPartitionedCallў
'stream_1_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_1_maxpool_1/PartitionedCall:output:0(^stream_2_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_22589412)
'stream_1_drop_1/StatefulPartitionedCallў
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_0_maxpool_1/PartitionedCall:output:0(^stream_1_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ъ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_22589182)
'stream_0_drop_1/StatefulPartitionedCall≤
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
GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_22586532*
(global_average_pooling1d/PartitionedCallЄ
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
GPU2*0J 8В *`
f[RY
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_22586602,
*global_average_pooling1d_1/PartitionedCallЄ
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
GPU2*0J 8В *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_22586672,
*global_average_pooling1d_2/PartitionedCallщ
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
GPU2*0J 8В *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_22586772
concatenate/PartitionedCallМ
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
GPU2*0J 8В *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_22588722!
dense_1_dropout/PartitionedCallЈ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_2259402dense_1_2259404*
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
D__inference_dense_1_layer_call_and_return_conditional_losses_22587112!
dense_1/StatefulPartitionedCallљ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_2259407batch_normalization_3_2259409batch_normalization_3_2259411batch_normalization_3_2259413*
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22582742/
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
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_22587302$
"dense_activation_1/PartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const 
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_2259356*"
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
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_1_2259356*"
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
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Const 
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_1_conv_1_2259351*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs≠
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1ў
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_1_conv_1/kernel/Regularizer/mulў
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/add–
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_2259351*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square≠
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2а
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xд
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1Ў
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1Щ
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Const 
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_2259346*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs≠
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1ў
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_2_conv_1/kernel/Regularizer/mulў
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/add–
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_2_conv_1_2259346*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square≠
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2а
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xд
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1Ў
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Constѓ
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_2259402*
_output_shapes
:	ј@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2 
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
dense_1/kernel/Regularizer/addµ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_2259402*
_output_shapes
:	ј@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2#
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

Identityр
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall(^stream_1_conv_1/StatefulPartitionedCall6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_1_drop_1/StatefulPartitionedCall,^stream_1_input_drop/StatefulPartitionedCall(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_2_drop_1/StatefulPartitionedCall,^stream_2_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:€€€€€€€€€ф:€€€€€€€€€ф:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:€€€€€€€€€ф
 
_user_specified_nameinputs:TP
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs:TP
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Ч{
э

D__inference_model_1_layer_call_and_return_conditional_losses_2260417
left_inputs'
basemodel_2260307:@
basemodel_2260309:@'
basemodel_2260311:@
basemodel_2260313:@'
basemodel_2260315:@
basemodel_2260317:@
basemodel_2260319:@
basemodel_2260321:@
basemodel_2260323:@
basemodel_2260325:@
basemodel_2260327:@
basemodel_2260329:@
basemodel_2260331:@
basemodel_2260333:@
basemodel_2260335:@
basemodel_2260337:@
basemodel_2260339:@
basemodel_2260341:@$
basemodel_2260343:	ј@
basemodel_2260345:@
basemodel_2260347:@
basemodel_2260349:@
basemodel_2260351:@
basemodel_2260353:@
identityИҐ!basemodel/StatefulPartitionedCallҐ-dense_1/kernel/Regularizer/Abs/ReadVariableOpҐ0dense_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpҐ5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpҐ8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpО
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsleft_inputsleft_inputsbasemodel_2260307basemodel_2260309basemodel_2260311basemodel_2260313basemodel_2260315basemodel_2260317basemodel_2260319basemodel_2260321basemodel_2260323basemodel_2260325basemodel_2260327basemodel_2260329basemodel_2260331basemodel_2260333basemodel_2260335basemodel_2260337basemodel_2260339basemodel_2260341basemodel_2260343basemodel_2260345basemodel_2260347basemodel_2260349basemodel_2260351basemodel_2260353*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_22587932#
!basemodel/StatefulPartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Constƒ
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_2260315*"
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
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_2260315*"
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
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Constƒ
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_2260311*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Abs≠
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1ў
&stream_1_conv_1/kernel/Regularizer/SumSum*stream_1_conv_1/kernel/Regularizer/Abs:y:03stream_1_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_1_conv_1/kernel/Regularizer/mulў
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/add 
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_2260311*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Square≠
*stream_1_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_2а
(stream_1_conv_1/kernel/Regularizer/Sum_1Sum-stream_1_conv_1/kernel/Regularizer/Square:y:03stream_1_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xд
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1Ў
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1Щ
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Constƒ
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_2260307*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp√
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Abs≠
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1ў
&stream_2_conv_1/kernel/Regularizer/SumSum*stream_2_conv_1/kernel/Regularizer/Abs:y:03stream_2_conv_1/kernel/Regularizer/Const_1:output:0*
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
&stream_2_conv_1/kernel/Regularizer/mulў
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/add 
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_2260307*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpѕ
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Square≠
*stream_2_conv_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_2а
(stream_2_conv_1/kernel/Regularizer/Sum_1Sum-stream_2_conv_1/kernel/Regularizer/Square:y:03stream_2_conv_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xд
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1Ў
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1Й
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const±
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_2260343*
_output_shapes
:	ј@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOp®
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2 
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
dense_1/kernel/Regularizer/addЈ
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_2260343*
_output_shapes
:	ј@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOpі
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ј@2#
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
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 2F
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
:€€€€€€€€€ф
%
_user_specified_nameleft_inputs
С	
“
7__inference_batch_normalization_1_layer_call_fn_2262323

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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22577342
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
п
“
7__inference_batch_normalization_2_layer_call_fn_2262522

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
 *,
_output_shapes
:€€€€€€€€€ф@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22591692
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€ф@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ф@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ф@
 
_user_specified_nameinputs
Щ
в
)__inference_model_1_layer_call_fn_2260751

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

unknown_17:	ј@

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
	
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_22602002
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
_construction_contextkEagerRuntime*[
_input_shapesJ
H:€€€€€€€€€ф: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
э
j
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_2262915

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
”
M
1__inference_dense_1_dropout_layer_call_fn_2262905

inputs
identityќ
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
GPU2*0J 8В *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_22586842
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
Й
s
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_2258667

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
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ъ@:T P
,
_output_shapes
:€€€€€€€€€ъ@
 
_user_specified_nameinputs
Л
h
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_2258872

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
Q
5__inference_stream_0_input_drop_layer_call_fn_2261912

inputs
identity÷
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_22583822
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ф:T P
,
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs"®L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*є
serving_default•
H
left_inputs9
serving_default_left_inputs:0€€€€€€€€€ф=
	basemodel0
StatefulPartitionedCall:0€€€€€€€€€@tensorflow/serving/predict:№ю
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
є__call__
+Ї&call_and_return_all_conditional_losses
ї_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
Е
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
%trainable_variables
&	variables
'regularization_losses
(	keras_api
Љ__call__
+љ&call_and_return_all_conditional_losses"
_tf_keras_network
Ц
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
÷
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
 "
trackable_list_wrapper
ќ

Alayers
trainable_variables
Blayer_regularization_losses
Cmetrics
	variables
regularization_losses
Dnon_trainable_variables
Elayer_metrics
є__call__
ї_default_save_signature
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
-
Њserving_default"
signature_map
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
І
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
њ__call__
+ј&call_and_return_all_conditional_losses"
_tf_keras_layer
І
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
Ѕ__call__
+¬&call_and_return_all_conditional_losses"
_tf_keras_layer
І
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
√__call__
+ƒ&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

)kernel
*bias
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
≈__call__
+∆&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

+kernel
,bias
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
«__call__
+»&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

-kernel
.bias
Ztrainable_variables
[	variables
\regularization_losses
]	keras_api
…__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
м
^axis
	/gamma
0beta
9moving_mean
:moving_variance
_trainable_variables
`	variables
aregularization_losses
b	keras_api
Ћ__call__
+ћ&call_and_return_all_conditional_losses"
_tf_keras_layer
м
caxis
	1gamma
2beta
;moving_mean
<moving_variance
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
Ќ__call__
+ќ&call_and_return_all_conditional_losses"
_tf_keras_layer
м
haxis
	3gamma
4beta
=moving_mean
>moving_variance
itrainable_variables
j	variables
kregularization_losses
l	keras_api
ѕ__call__
+–&call_and_return_all_conditional_losses"
_tf_keras_layer
І
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
—__call__
+“&call_and_return_all_conditional_losses"
_tf_keras_layer
І
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
”__call__
+‘&call_and_return_all_conditional_losses"
_tf_keras_layer
І
utrainable_variables
v	variables
wregularization_losses
x	keras_api
’__call__
+÷&call_and_return_all_conditional_losses"
_tf_keras_layer
І
ytrainable_variables
z	variables
{regularization_losses
|	keras_api
„__call__
+Ў&call_and_return_all_conditional_losses"
_tf_keras_layer
®
}trainable_variables
~	variables
regularization_losses
А	keras_api
ў__call__
+Џ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Бtrainable_variables
В	variables
Гregularization_losses
Д	keras_api
џ__call__
+№&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Еtrainable_variables
Ж	variables
Зregularization_losses
И	keras_api
Ё__call__
+ё&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Йtrainable_variables
К	variables
Лregularization_losses
М	keras_api
я__call__
+а&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Нtrainable_variables
О	variables
Пregularization_losses
Р	keras_api
б__call__
+в&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Сtrainable_variables
Т	variables
Уregularization_losses
Ф	keras_api
г__call__
+д&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Хtrainable_variables
Ц	variables
Чregularization_losses
Ш	keras_api
е__call__
+ж&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Щtrainable_variables
Ъ	variables
Ыregularization_losses
Ь	keras_api
з__call__
+и&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Эtrainable_variables
Ю	variables
Яregularization_losses
†	keras_api
й__call__
+к&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
°trainable_variables
Ґ	variables
£regularization_losses
§	keras_api
л__call__
+м&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ

5kernel
6bias
•trainable_variables
¶	variables
Іregularization_losses
®	keras_api
н__call__
+о&call_and_return_all_conditional_losses"
_tf_keras_layer
с
	©axis
	7gamma
8beta
?moving_mean
@moving_variance
™trainable_variables
Ђ	variables
ђregularization_losses
≠	keras_api
п__call__
+р&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Ѓtrainable_variables
ѓ	variables
∞regularization_losses
±	keras_api
с__call__
+т&call_and_return_all_conditional_losses"
_tf_keras_layer
Ц
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
÷
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
@
у0
ф1
х2
ц3"
trackable_list_wrapper
µ
≤layers
%trainable_variables
 ≥layer_regularization_losses
іmetrics
&	variables
'regularization_losses
µnon_trainable_variables
ґlayer_metrics
Љ__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
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
!:	ј@2dense_1/kernel
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
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Јlayers
Ftrainable_variables
 Єlayer_regularization_losses
єmetrics
G	variables
Hregularization_losses
Їnon_trainable_variables
їlayer_metrics
њ__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Љlayers
Jtrainable_variables
 љlayer_regularization_losses
Њmetrics
K	variables
Lregularization_losses
њnon_trainable_variables
јlayer_metrics
Ѕ__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ѕlayers
Ntrainable_variables
 ¬layer_regularization_losses
√metrics
O	variables
Pregularization_losses
ƒnon_trainable_variables
≈layer_metrics
√__call__
+ƒ&call_and_return_all_conditional_losses
'ƒ"call_and_return_conditional_losses"
_generic_user_object
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
(
у0"
trackable_list_wrapper
µ
∆layers
Rtrainable_variables
 «layer_regularization_losses
»metrics
S	variables
Tregularization_losses
…non_trainable_variables
 layer_metrics
≈__call__
+∆&call_and_return_all_conditional_losses
'∆"call_and_return_conditional_losses"
_generic_user_object
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
(
ф0"
trackable_list_wrapper
µ
Ћlayers
Vtrainable_variables
 ћlayer_regularization_losses
Ќmetrics
W	variables
Xregularization_losses
ќnon_trainable_variables
ѕlayer_metrics
«__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
(
х0"
trackable_list_wrapper
µ
–layers
Ztrainable_variables
 —layer_regularization_losses
“metrics
[	variables
\regularization_losses
”non_trainable_variables
‘layer_metrics
…__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
µ
’layers
_trainable_variables
 ÷layer_regularization_losses
„metrics
`	variables
aregularization_losses
Ўnon_trainable_variables
ўlayer_metrics
Ћ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
µ
Џlayers
dtrainable_variables
 џlayer_regularization_losses
№metrics
e	variables
fregularization_losses
Ёnon_trainable_variables
ёlayer_metrics
Ќ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
µ
яlayers
itrainable_variables
 аlayer_regularization_losses
бmetrics
j	variables
kregularization_losses
вnon_trainable_variables
гlayer_metrics
ѕ__call__
+–&call_and_return_all_conditional_losses
'–"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
дlayers
mtrainable_variables
 еlayer_regularization_losses
жmetrics
n	variables
oregularization_losses
зnon_trainable_variables
иlayer_metrics
—__call__
+“&call_and_return_all_conditional_losses
'“"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
йlayers
qtrainable_variables
 кlayer_regularization_losses
лmetrics
r	variables
sregularization_losses
мnon_trainable_variables
нlayer_metrics
”__call__
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
оlayers
utrainable_variables
 пlayer_regularization_losses
рmetrics
v	variables
wregularization_losses
сnon_trainable_variables
тlayer_metrics
’__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
уlayers
ytrainable_variables
 фlayer_regularization_losses
хmetrics
z	variables
{regularization_losses
цnon_trainable_variables
чlayer_metrics
„__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
шlayers
}trainable_variables
 щlayer_regularization_losses
ъmetrics
~	variables
regularization_losses
ыnon_trainable_variables
ьlayer_metrics
ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
эlayers
Бtrainable_variables
 юlayer_regularization_losses
€metrics
В	variables
Гregularization_losses
Аnon_trainable_variables
Бlayer_metrics
џ__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Вlayers
Еtrainable_variables
 Гlayer_regularization_losses
Дmetrics
Ж	variables
Зregularization_losses
Еnon_trainable_variables
Жlayer_metrics
Ё__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Зlayers
Йtrainable_variables
 Иlayer_regularization_losses
Йmetrics
К	variables
Лregularization_losses
Кnon_trainable_variables
Лlayer_metrics
я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Мlayers
Нtrainable_variables
 Нlayer_regularization_losses
Оmetrics
О	variables
Пregularization_losses
Пnon_trainable_variables
Рlayer_metrics
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
Є
Сlayers
Сtrainable_variables
 Тlayer_regularization_losses
Уmetrics
Т	variables
Уregularization_losses
Фnon_trainable_variables
Хlayer_metrics
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
Є
Цlayers
Хtrainable_variables
 Чlayer_regularization_losses
Шmetrics
Ц	variables
Чregularization_losses
Щnon_trainable_variables
Ъlayer_metrics
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ыlayers
Щtrainable_variables
 Ьlayer_regularization_losses
Эmetrics
Ъ	variables
Ыregularization_losses
Юnon_trainable_variables
Яlayer_metrics
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
†layers
Эtrainable_variables
 °layer_regularization_losses
Ґmetrics
Ю	variables
Яregularization_losses
£non_trainable_variables
§layer_metrics
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
•layers
°trainable_variables
 ¶layer_regularization_losses
Іmetrics
Ґ	variables
£regularization_losses
®non_trainable_variables
©layer_metrics
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
(
ц0"
trackable_list_wrapper
Є
™layers
•trainable_variables
 Ђlayer_regularization_losses
ђmetrics
¶	variables
Іregularization_losses
≠non_trainable_variables
Ѓlayer_metrics
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
Є
ѓlayers
™trainable_variables
 ∞layer_regularization_losses
±metrics
Ђ	variables
ђregularization_losses
≤non_trainable_variables
≥layer_metrics
п__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
іlayers
Ѓtrainable_variables
 µlayer_regularization_losses
ґmetrics
ѓ	variables
∞regularization_losses
Јnon_trainable_variables
Єlayer_metrics
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
ю
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
у0"
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
ф0"
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
х0"
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
90
:1"
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
;0
<1"
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
=0
>1"
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
(
ц0"
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
?0
@1"
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
)__inference_model_1_layer_call_fn_2260032
)__inference_model_1_layer_call_fn_2260698
)__inference_model_1_layer_call_fn_2260751
)__inference_model_1_layer_call_fn_2260304ј
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
D__inference_model_1_layer_call_and_return_conditional_losses_2260948
D__inference_model_1_layer_call_and_return_conditional_losses_2261242
D__inference_model_1_layer_call_and_return_conditional_losses_2260417
D__inference_model_1_layer_call_and_return_conditional_losses_2260530ј
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
"__inference__wrapped_model_2257548left_inputs"Ш
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
ъ2ч
+__inference_basemodel_layer_call_fn_2258844
+__inference_basemodel_layer_call_fn_2261357
+__inference_basemodel_layer_call_fn_2261412
+__inference_basemodel_layer_call_fn_2259584ј
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
ж2г
F__inference_basemodel_layer_call_and_return_conditional_losses_2261611
F__inference_basemodel_layer_call_and_return_conditional_losses_2261907
F__inference_basemodel_layer_call_and_return_conditional_losses_2259724
F__inference_basemodel_layer_call_and_return_conditional_losses_2259864ј
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
%__inference_signature_wrapper_2260645left_inputs"Ф
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
5__inference_stream_0_input_drop_layer_call_fn_2261912
5__inference_stream_0_input_drop_layer_call_fn_2261917і
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
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_2261922
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_2261934і
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
®2•
5__inference_stream_1_input_drop_layer_call_fn_2261939
5__inference_stream_1_input_drop_layer_call_fn_2261944і
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
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_2261949
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_2261961і
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
®2•
5__inference_stream_2_input_drop_layer_call_fn_2261966
5__inference_stream_2_input_drop_layer_call_fn_2261971і
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
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_2261976
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_2261988і
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
1__inference_stream_0_conv_1_layer_call_fn_2262012Ґ
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
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_2262042Ґ
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
џ2Ў
1__inference_stream_1_conv_1_layer_call_fn_2262066Ґ
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
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_2262096Ґ
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
џ2Ў
1__inference_stream_2_conv_1_layer_call_fn_2262120Ґ
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
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_2262150Ґ
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
5__inference_batch_normalization_layer_call_fn_2262163
5__inference_batch_normalization_layer_call_fn_2262176
5__inference_batch_normalization_layer_call_fn_2262189
5__inference_batch_normalization_layer_call_fn_2262202і
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
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2262222
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2262256
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2262276
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2262310і
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
Ю2Ы
7__inference_batch_normalization_1_layer_call_fn_2262323
7__inference_batch_normalization_1_layer_call_fn_2262336
7__inference_batch_normalization_1_layer_call_fn_2262349
7__inference_batch_normalization_1_layer_call_fn_2262362і
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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2262382
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2262416
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2262436
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2262470і
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
Ю2Ы
7__inference_batch_normalization_2_layer_call_fn_2262483
7__inference_batch_normalization_2_layer_call_fn_2262496
7__inference_batch_normalization_2_layer_call_fn_2262509
7__inference_batch_normalization_2_layer_call_fn_2262522і
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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2262542
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2262576
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2262596
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2262630і
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
,__inference_activation_layer_call_fn_2262635Ґ
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
G__inference_activation_layer_call_and_return_conditional_losses_2262640Ґ
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
Ў2’
.__inference_activation_1_layer_call_fn_2262645Ґ
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
I__inference_activation_1_layer_call_and_return_conditional_losses_2262650Ґ
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
Ў2’
.__inference_activation_2_layer_call_fn_2262655Ґ
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
I__inference_activation_2_layer_call_and_return_conditional_losses_2262660Ґ
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
4__inference_stream_0_maxpool_1_layer_call_fn_2262665
4__inference_stream_0_maxpool_1_layer_call_fn_2262670Ґ
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
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_2262678
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_2262686Ґ
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
4__inference_stream_1_maxpool_1_layer_call_fn_2262691
4__inference_stream_1_maxpool_1_layer_call_fn_2262696Ґ
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
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_2262704
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_2262712Ґ
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
4__inference_stream_2_maxpool_1_layer_call_fn_2262717
4__inference_stream_2_maxpool_1_layer_call_fn_2262722Ґ
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
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_2262730
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_2262738Ґ
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
1__inference_stream_0_drop_1_layer_call_fn_2262743
1__inference_stream_0_drop_1_layer_call_fn_2262748і
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
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_2262753
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_2262765і
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
†2Э
1__inference_stream_1_drop_1_layer_call_fn_2262770
1__inference_stream_1_drop_1_layer_call_fn_2262775і
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
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_2262780
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_2262792і
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
†2Э
1__inference_stream_2_drop_1_layer_call_fn_2262797
1__inference_stream_2_drop_1_layer_call_fn_2262802і
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
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_2262807
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_2262819і
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
:__inference_global_average_pooling1d_layer_call_fn_2262824
:__inference_global_average_pooling1d_layer_call_fn_2262829ѓ
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
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_2262835
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_2262841ѓ
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
±2Ѓ
<__inference_global_average_pooling1d_1_layer_call_fn_2262846
<__inference_global_average_pooling1d_1_layer_call_fn_2262851ѓ
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
з2д
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_2262857
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_2262863ѓ
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
±2Ѓ
<__inference_global_average_pooling1d_2_layer_call_fn_2262868
<__inference_global_average_pooling1d_2_layer_call_fn_2262873ѓ
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
з2д
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_2262879
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_2262885ѓ
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
„2‘
-__inference_concatenate_layer_call_fn_2262892Ґ
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
H__inference_concatenate_layer_call_and_return_conditional_losses_2262900Ґ
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
1__inference_dense_1_dropout_layer_call_fn_2262905
1__inference_dense_1_dropout_layer_call_fn_2262910і
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
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_2262915
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_2262919і
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
)__inference_dense_1_layer_call_fn_2262943Ґ
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
D__inference_dense_1_layer_call_and_return_conditional_losses_2262968Ґ
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
7__inference_batch_normalization_3_layer_call_fn_2262981
7__inference_batch_normalization_3_layer_call_fn_2262994і
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2263014
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2263048і
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
4__inference_dense_activation_1_layer_call_fn_2263053Ґ
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
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_2263057Ґ
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
__inference_loss_fn_0_2263077П
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
__inference_loss_fn_1_2263097П
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
__inference_loss_fn_2_2263117П
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
__inference_loss_fn_3_2263137П
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
annotations™ *Ґ ≥
"__inference__wrapped_model_2257548М-.+,)*>3=4<1;2:/9056@7?89Ґ6
/Ґ,
*К'
left_inputs€€€€€€€€€ф
™ "5™2
0
	basemodel#К 
	basemodel€€€€€€€€€@ѓ
I__inference_activation_1_layer_call_and_return_conditional_losses_2262650b4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ф@
™ "*Ґ'
 К
0€€€€€€€€€ф@
Ъ З
.__inference_activation_1_layer_call_fn_2262645U4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ф@
™ "К€€€€€€€€€ф@ѓ
I__inference_activation_2_layer_call_and_return_conditional_losses_2262660b4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ф@
™ "*Ґ'
 К
0€€€€€€€€€ф@
Ъ З
.__inference_activation_2_layer_call_fn_2262655U4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ф@
™ "К€€€€€€€€€ф@≠
G__inference_activation_layer_call_and_return_conditional_losses_2262640b4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ф@
™ "*Ґ'
 К
0€€€€€€€€€ф@
Ъ Е
,__inference_activation_layer_call_fn_2262635U4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ф@
™ "К€€€€€€€€€ф@І
F__inference_basemodel_layer_call_and_return_conditional_losses_2259724№-.+,)*>3=4<1;2:/9056@7?8ШҐФ
МҐИ
~Ъ{
'К$
inputs_0€€€€€€€€€ф
'К$
inputs_1€€€€€€€€€ф
'К$
inputs_2€€€€€€€€€ф
p 

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ І
F__inference_basemodel_layer_call_and_return_conditional_losses_2259864№-.+,)*=>34;<129:/056?@78ШҐФ
МҐИ
~Ъ{
'К$
inputs_0€€€€€€€€€ф
'К$
inputs_1€€€€€€€€€ф
'К$
inputs_2€€€€€€€€€ф
p

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ І
F__inference_basemodel_layer_call_and_return_conditional_losses_2261611№-.+,)*>3=4<1;2:/9056@7?8ШҐФ
МҐИ
~Ъ{
'К$
inputs/0€€€€€€€€€ф
'К$
inputs/1€€€€€€€€€ф
'К$
inputs/2€€€€€€€€€ф
p 

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ І
F__inference_basemodel_layer_call_and_return_conditional_losses_2261907№-.+,)*=>34;<129:/056?@78ШҐФ
МҐИ
~Ъ{
'К$
inputs/0€€€€€€€€€ф
'К$
inputs/1€€€€€€€€€ф
'К$
inputs/2€€€€€€€€€ф
p

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ €
+__inference_basemodel_layer_call_fn_2258844ѕ-.+,)*>3=4<1;2:/9056@7?8ШҐФ
МҐИ
~Ъ{
'К$
inputs_0€€€€€€€€€ф
'К$
inputs_1€€€€€€€€€ф
'К$
inputs_2€€€€€€€€€ф
p 

 
™ "К€€€€€€€€€@€
+__inference_basemodel_layer_call_fn_2259584ѕ-.+,)*=>34;<129:/056?@78ШҐФ
МҐИ
~Ъ{
'К$
inputs_0€€€€€€€€€ф
'К$
inputs_1€€€€€€€€€ф
'К$
inputs_2€€€€€€€€€ф
p

 
™ "К€€€€€€€€€@€
+__inference_basemodel_layer_call_fn_2261357ѕ-.+,)*>3=4<1;2:/9056@7?8ШҐФ
МҐИ
~Ъ{
'К$
inputs/0€€€€€€€€€ф
'К$
inputs/1€€€€€€€€€ф
'К$
inputs/2€€€€€€€€€ф
p 

 
™ "К€€€€€€€€€@€
+__inference_basemodel_layer_call_fn_2261412ѕ-.+,)*=>34;<129:/056?@78ШҐФ
МҐИ
~Ъ{
'К$
inputs/0€€€€€€€€€ф
'К$
inputs/1€€€€€€€€€ф
'К$
inputs/2€€€€€€€€€ф
p

 
™ "К€€€€€€€€€@“
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2262382|<1;2@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ “
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2262416|;<12@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ ¬
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2262436l<1;28Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф@
p 
™ "*Ґ'
 К
0€€€€€€€€€ф@
Ъ ¬
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2262470l;<128Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф@
p
™ "*Ґ'
 К
0€€€€€€€€€ф@
Ъ ™
7__inference_batch_normalization_1_layer_call_fn_2262323o<1;2@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 
™ "%К"€€€€€€€€€€€€€€€€€€@™
7__inference_batch_normalization_1_layer_call_fn_2262336o;<12@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p
™ "%К"€€€€€€€€€€€€€€€€€€@Ъ
7__inference_batch_normalization_1_layer_call_fn_2262349_<1;28Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф@
p 
™ "К€€€€€€€€€ф@Ъ
7__inference_batch_normalization_1_layer_call_fn_2262362_;<128Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф@
p
™ "К€€€€€€€€€ф@“
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2262542|>3=4@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ “
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2262576|=>34@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ ¬
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2262596l>3=48Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф@
p 
™ "*Ґ'
 К
0€€€€€€€€€ф@
Ъ ¬
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2262630l=>348Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф@
p
™ "*Ґ'
 К
0€€€€€€€€€ф@
Ъ ™
7__inference_batch_normalization_2_layer_call_fn_2262483o>3=4@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 
™ "%К"€€€€€€€€€€€€€€€€€€@™
7__inference_batch_normalization_2_layer_call_fn_2262496o=>34@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p
™ "%К"€€€€€€€€€€€€€€€€€€@Ъ
7__inference_batch_normalization_2_layer_call_fn_2262509_>3=48Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф@
p 
™ "К€€€€€€€€€ф@Ъ
7__inference_batch_normalization_2_layer_call_fn_2262522_=>348Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф@
p
™ "К€€€€€€€€€ф@Є
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2263014b@7?83Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Є
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2263048b?@783Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Р
7__inference_batch_normalization_3_layer_call_fn_2262981U@7?83Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@Р
7__inference_batch_normalization_3_layer_call_fn_2262994U?@783Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@–
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2262222|:/90@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ –
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2262256|9:/0@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€@
Ъ ј
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2262276l:/908Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф@
p 
™ "*Ґ'
 К
0€€€€€€€€€ф@
Ъ ј
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2262310l9:/08Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф@
p
™ "*Ґ'
 К
0€€€€€€€€€ф@
Ъ ®
5__inference_batch_normalization_layer_call_fn_2262163o:/90@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p 
™ "%К"€€€€€€€€€€€€€€€€€€@®
5__inference_batch_normalization_layer_call_fn_2262176o9:/0@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€@
p
™ "%К"€€€€€€€€€€€€€€€€€€@Ш
5__inference_batch_normalization_layer_call_fn_2262189_:/908Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф@
p 
™ "К€€€€€€€€€ф@Ш
5__inference_batch_normalization_layer_call_fn_2262202_9:/08Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф@
p
™ "К€€€€€€€€€ф@х
H__inference_concatenate_layer_call_and_return_conditional_losses_2262900®~Ґ{
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
Ъ Ќ
-__inference_concatenate_layer_call_fn_2262892Ы~Ґ{
tҐq
oЪl
"К
inputs/0€€€€€€€€€@
"К
inputs/1€€€€€€€€€@
"К
inputs/2€€€€€€€€€@
™ "К€€€€€€€€€јЃ
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_2262915^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ј
p 
™ "&Ґ#
К
0€€€€€€€€€ј
Ъ Ѓ
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_2262919^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ј
p
™ "&Ґ#
К
0€€€€€€€€€ј
Ъ Ж
1__inference_dense_1_dropout_layer_call_fn_2262905Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ј
p 
™ "К€€€€€€€€€јЖ
1__inference_dense_1_dropout_layer_call_fn_2262910Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ј
p
™ "К€€€€€€€€€ј•
D__inference_dense_1_layer_call_and_return_conditional_losses_2262968]560Ґ-
&Ґ#
!К
inputs€€€€€€€€€ј
™ "%Ґ"
К
0€€€€€€€€€@
Ъ }
)__inference_dense_1_layer_call_fn_2262943P560Ґ-
&Ґ#
!К
inputs€€€€€€€€€ј
™ "К€€€€€€€€€@Ђ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_2263057X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Г
4__inference_dense_activation_1_layer_call_fn_2263053K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€@÷
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_2262857{IҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ ".Ґ+
$К!
0€€€€€€€€€€€€€€€€€€
Ъ Љ
W__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_2262863a8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ъ@

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Ѓ
<__inference_global_average_pooling1d_1_layer_call_fn_2262846nIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ "!К€€€€€€€€€€€€€€€€€€Ф
<__inference_global_average_pooling1d_1_layer_call_fn_2262851T8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ъ@

 
™ "К€€€€€€€€€@÷
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_2262879{IҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ ".Ґ+
$К!
0€€€€€€€€€€€€€€€€€€
Ъ Љ
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_2262885a8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ъ@

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Ѓ
<__inference_global_average_pooling1d_2_layer_call_fn_2262868nIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ "!К€€€€€€€€€€€€€€€€€€Ф
<__inference_global_average_pooling1d_2_layer_call_fn_2262873T8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ъ@

 
™ "К€€€€€€€€€@‘
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_2262835{IҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ ".Ґ+
$К!
0€€€€€€€€€€€€€€€€€€
Ъ Ї
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_2262841a8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ъ@

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ђ
:__inference_global_average_pooling1d_layer_call_fn_2262824nIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ "!К€€€€€€€€€€€€€€€€€€Т
:__inference_global_average_pooling1d_layer_call_fn_2262829T8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ъ@

 
™ "К€€€€€€€€€@<
__inference_loss_fn_0_2263077)Ґ

Ґ 
™ "К <
__inference_loss_fn_1_2263097+Ґ

Ґ 
™ "К <
__inference_loss_fn_2_2263117-Ґ

Ґ 
™ "К <
__inference_loss_fn_3_22631375Ґ

Ґ 
™ "К Ќ
D__inference_model_1_layer_call_and_return_conditional_losses_2260417Д-.+,)*>3=4<1;2:/9056@7?8AҐ>
7Ґ4
*К'
left_inputs€€€€€€€€€ф
p 

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Ќ
D__inference_model_1_layer_call_and_return_conditional_losses_2260530Д-.+,)*=>34;<129:/056?@78AҐ>
7Ґ4
*К'
left_inputs€€€€€€€€€ф
p

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ «
D__inference_model_1_layer_call_and_return_conditional_losses_2260948-.+,)*>3=4<1;2:/9056@7?8<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€ф
p 

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ «
D__inference_model_1_layer_call_and_return_conditional_losses_2261242-.+,)*=>34;<129:/056?@78<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€ф
p

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ §
)__inference_model_1_layer_call_fn_2260032w-.+,)*>3=4<1;2:/9056@7?8AҐ>
7Ґ4
*К'
left_inputs€€€€€€€€€ф
p 

 
™ "К€€€€€€€€€@§
)__inference_model_1_layer_call_fn_2260304w-.+,)*=>34;<129:/056?@78AҐ>
7Ґ4
*К'
left_inputs€€€€€€€€€ф
p

 
™ "К€€€€€€€€€@Я
)__inference_model_1_layer_call_fn_2260698r-.+,)*>3=4<1;2:/9056@7?8<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€ф
p 

 
™ "К€€€€€€€€€@Я
)__inference_model_1_layer_call_fn_2260751r-.+,)*=>34;<129:/056?@78<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€ф
p

 
™ "К€€€€€€€€€@≈
%__inference_signature_wrapper_2260645Ы-.+,)*>3=4<1;2:/9056@7?8HҐE
Ґ 
>™;
9
left_inputs*К'
left_inputs€€€€€€€€€ф"5™2
0
	basemodel#К 
	basemodel€€€€€€€€€@ґ
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_2262042f)*4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ф
™ "*Ґ'
 К
0€€€€€€€€€ф@
Ъ О
1__inference_stream_0_conv_1_layer_call_fn_2262012Y)*4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ф
™ "К€€€€€€€€€ф@ґ
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_2262753f8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ъ@
p 
™ "*Ґ'
 К
0€€€€€€€€€ъ@
Ъ ґ
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_2262765f8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ъ@
p
™ "*Ґ'
 К
0€€€€€€€€€ъ@
Ъ О
1__inference_stream_0_drop_1_layer_call_fn_2262743Y8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ъ@
p 
™ "К€€€€€€€€€ъ@О
1__inference_stream_0_drop_1_layer_call_fn_2262748Y8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ъ@
p
™ "К€€€€€€€€€ъ@Ї
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_2261922f8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф
p 
™ "*Ґ'
 К
0€€€€€€€€€ф
Ъ Ї
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_2261934f8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф
p
™ "*Ґ'
 К
0€€€€€€€€€ф
Ъ Т
5__inference_stream_0_input_drop_layer_call_fn_2261912Y8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф
p 
™ "К€€€€€€€€€фТ
5__inference_stream_0_input_drop_layer_call_fn_2261917Y8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф
p
™ "К€€€€€€€€€фЎ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_2262678ДEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";Ґ8
1К.
0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ µ
O__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_2262686b4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ф@
™ "*Ґ'
 К
0€€€€€€€€€ъ@
Ъ ѓ
4__inference_stream_0_maxpool_1_layer_call_fn_2262665wEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€Н
4__inference_stream_0_maxpool_1_layer_call_fn_2262670U4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ф@
™ "К€€€€€€€€€ъ@ґ
L__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_2262096f+,4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ф
™ "*Ґ'
 К
0€€€€€€€€€ф@
Ъ О
1__inference_stream_1_conv_1_layer_call_fn_2262066Y+,4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ф
™ "К€€€€€€€€€ф@ґ
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_2262780f8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ъ@
p 
™ "*Ґ'
 К
0€€€€€€€€€ъ@
Ъ ґ
L__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_2262792f8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ъ@
p
™ "*Ґ'
 К
0€€€€€€€€€ъ@
Ъ О
1__inference_stream_1_drop_1_layer_call_fn_2262770Y8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ъ@
p 
™ "К€€€€€€€€€ъ@О
1__inference_stream_1_drop_1_layer_call_fn_2262775Y8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ъ@
p
™ "К€€€€€€€€€ъ@Ї
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_2261949f8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф
p 
™ "*Ґ'
 К
0€€€€€€€€€ф
Ъ Ї
P__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_2261961f8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф
p
™ "*Ґ'
 К
0€€€€€€€€€ф
Ъ Т
5__inference_stream_1_input_drop_layer_call_fn_2261939Y8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф
p 
™ "К€€€€€€€€€фТ
5__inference_stream_1_input_drop_layer_call_fn_2261944Y8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф
p
™ "К€€€€€€€€€фЎ
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_2262704ДEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";Ґ8
1К.
0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ µ
O__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_2262712b4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ф@
™ "*Ґ'
 К
0€€€€€€€€€ъ@
Ъ ѓ
4__inference_stream_1_maxpool_1_layer_call_fn_2262691wEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€Н
4__inference_stream_1_maxpool_1_layer_call_fn_2262696U4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ф@
™ "К€€€€€€€€€ъ@ґ
L__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_2262150f-.4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ф
™ "*Ґ'
 К
0€€€€€€€€€ф@
Ъ О
1__inference_stream_2_conv_1_layer_call_fn_2262120Y-.4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ф
™ "К€€€€€€€€€ф@ґ
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_2262807f8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ъ@
p 
™ "*Ґ'
 К
0€€€€€€€€€ъ@
Ъ ґ
L__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_2262819f8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ъ@
p
™ "*Ґ'
 К
0€€€€€€€€€ъ@
Ъ О
1__inference_stream_2_drop_1_layer_call_fn_2262797Y8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ъ@
p 
™ "К€€€€€€€€€ъ@О
1__inference_stream_2_drop_1_layer_call_fn_2262802Y8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ъ@
p
™ "К€€€€€€€€€ъ@Ї
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_2261976f8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф
p 
™ "*Ґ'
 К
0€€€€€€€€€ф
Ъ Ї
P__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_2261988f8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф
p
™ "*Ґ'
 К
0€€€€€€€€€ф
Ъ Т
5__inference_stream_2_input_drop_layer_call_fn_2261966Y8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф
p 
™ "К€€€€€€€€€фТ
5__inference_stream_2_input_drop_layer_call_fn_2261971Y8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€ф
p
™ "К€€€€€€€€€фЎ
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_2262730ДEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";Ґ8
1К.
0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ µ
O__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_2262738b4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ф@
™ "*Ґ'
 К
0€€€€€€€€€ъ@
Ъ ѓ
4__inference_stream_2_maxpool_1_layer_call_fn_2262717wEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€Н
4__inference_stream_2_maxpool_1_layer_call_fn_2262722U4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ф@
™ "К€€€€€€€€€ъ@