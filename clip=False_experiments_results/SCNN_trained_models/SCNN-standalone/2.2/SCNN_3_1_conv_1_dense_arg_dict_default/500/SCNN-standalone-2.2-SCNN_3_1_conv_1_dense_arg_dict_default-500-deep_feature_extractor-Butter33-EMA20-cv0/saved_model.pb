▐■3
╨ж
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
 И"serve*2.6.22v2.6.1-9-gc2363d6d0258мш/
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
shape:	└@*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	└@*
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
shape:@*2
shared_name#!batch_normalization_3/moving_mean
У
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
в
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
dtype0*─b
value║bB╖b B░b
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
о
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
╢
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
н

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
а	keras_api
V
бtrainable_variables
в	variables
гregularization_losses
д	keras_api
l

5kernel
6bias
еtrainable_variables
ж	variables
зregularization_losses
и	keras_api
Ь
	йaxis
	7gamma
8beta
?moving_mean
@moving_variance
кtrainable_variables
л	variables
мregularization_losses
н	keras_api
V
оtrainable_variables
п	variables
░regularization_losses
▒	keras_api
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
╢
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
▓
▓layers
%trainable_variables
 │layer_regularization_losses
┤metrics
&	variables
'regularization_losses
╡non_trainable_variables
╢layer_metrics
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
▓
╖layers
Ftrainable_variables
 ╕layer_regularization_losses
╣metrics
G	variables
Hregularization_losses
║non_trainable_variables
╗layer_metrics
 
 
 
▓
╝layers
Jtrainable_variables
 ╜layer_regularization_losses
╛metrics
K	variables
Lregularization_losses
┐non_trainable_variables
└layer_metrics
 
 
 
▓
┴layers
Ntrainable_variables
 ┬layer_regularization_losses
├metrics
O	variables
Pregularization_losses
─non_trainable_variables
┼layer_metrics

)0
*1

)0
*1
 
▓
╞layers
Rtrainable_variables
 ╟layer_regularization_losses
╚metrics
S	variables
Tregularization_losses
╔non_trainable_variables
╩layer_metrics

+0
,1

+0
,1
 
▓
╦layers
Vtrainable_variables
 ╠layer_regularization_losses
═metrics
W	variables
Xregularization_losses
╬non_trainable_variables
╧layer_metrics

-0
.1

-0
.1
 
▓
╨layers
Ztrainable_variables
 ╤layer_regularization_losses
╥metrics
[	variables
\regularization_losses
╙non_trainable_variables
╘layer_metrics
 

/0
01

/0
01
92
:3
 
▓
╒layers
_trainable_variables
 ╓layer_regularization_losses
╫metrics
`	variables
aregularization_losses
╪non_trainable_variables
┘layer_metrics
 

10
21

10
21
;2
<3
 
▓
┌layers
dtrainable_variables
 █layer_regularization_losses
▄metrics
e	variables
fregularization_losses
▌non_trainable_variables
▐layer_metrics
 

30
41

30
41
=2
>3
 
▓
▀layers
itrainable_variables
 рlayer_regularization_losses
сmetrics
j	variables
kregularization_losses
тnon_trainable_variables
уlayer_metrics
 
 
 
▓
фlayers
mtrainable_variables
 хlayer_regularization_losses
цmetrics
n	variables
oregularization_losses
чnon_trainable_variables
шlayer_metrics
 
 
 
▓
щlayers
qtrainable_variables
 ъlayer_regularization_losses
ыmetrics
r	variables
sregularization_losses
ьnon_trainable_variables
эlayer_metrics
 
 
 
▓
юlayers
utrainable_variables
 яlayer_regularization_losses
Ёmetrics
v	variables
wregularization_losses
ёnon_trainable_variables
Єlayer_metrics
 
 
 
▓
єlayers
ytrainable_variables
 Їlayer_regularization_losses
їmetrics
z	variables
{regularization_losses
Ўnon_trainable_variables
ўlayer_metrics
 
 
 
▓
°layers
}trainable_variables
 ∙layer_regularization_losses
·metrics
~	variables
regularization_losses
√non_trainable_variables
№layer_metrics
 
 
 
╡
¤layers
Бtrainable_variables
 ■layer_regularization_losses
 metrics
В	variables
Гregularization_losses
Аnon_trainable_variables
Бlayer_metrics
 
 
 
╡
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
╡
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
╡
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
╡
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
╡
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
╡
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
╡
аlayers
Эtrainable_variables
 бlayer_regularization_losses
вmetrics
Ю	variables
Яregularization_losses
гnon_trainable_variables
дlayer_metrics
 
 
 
╡
еlayers
бtrainable_variables
 жlayer_regularization_losses
зmetrics
в	variables
гregularization_losses
иnon_trainable_variables
йlayer_metrics

50
61

50
61
 
╡
кlayers
еtrainable_variables
 лlayer_regularization_losses
мmetrics
ж	variables
зregularization_losses
нnon_trainable_variables
оlayer_metrics
 

70
81

70
81
?2
@3
 
╡
пlayers
кtrainable_variables
 ░layer_regularization_losses
▒metrics
л	variables
мregularization_losses
▓non_trainable_variables
│layer_metrics
 
 
 
╡
┤layers
оtrainable_variables
 ╡layer_regularization_losses
╢metrics
п	variables
░regularization_losses
╖non_trainable_variables
╕layer_metrics
▐
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
:         Ї*
dtype0*!
shape:         Ї
Х
StatefulPartitionedCallStatefulPartitionedCallserving_default_left_inputsstream_2_conv_1/kernelstream_2_conv_1/biasstream_1_conv_1/kernelstream_1_conv_1/biasstream_0_conv_1/kernelstream_0_conv_1/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/beta%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/beta#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense_1/kerneldense_1/bias%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/beta*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_692759
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
┬
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
GPU2*0J 8В *(
f#R!
__inference__traced_save_695346
▌
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
GPU2*0J 8В *+
f&R$
"__inference__traced_restore_695428у▓.
ш
b
F__inference_activation_layer_call_and_return_conditional_losses_694754

inputs
identityS
TanhTanhinputs*
T0*,
_output_shapes
:         Ї@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:         Ї@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
й:
А
__inference__traced_save_695346
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
▒: :@:@:@:@:@:@:@:@:@:@:@:@:	└@:@:@:@:@:@:@:@:@:@:@:@: 2(
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
:	└@: 
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
И
т
$__inference_signature_wrapper_692759
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

unknown_17:	└@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИвStatefulPartitionedCallД
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
:         @*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_6896622
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:         Ї
%
_user_specified_nameleft_inputs
∙,
Н
K__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_694264

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpy
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
:         Ї2
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
:         Ї@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2	
BiasAddЩ
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Const▐
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absн
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul┘
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addф
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Squareн
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
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1╪
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1p
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identity 
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Ї: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
И
r
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_694999

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
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
щ
╧
4__inference_batch_normalization_layer_call_fn_694316

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_6911632
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Ї@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
ъ
d
H__inference_activation_1_layer_call_and_return_conditional_losses_694764

inputs
identityS
TanhTanhinputs*
T0*,
_output_shapes
:         Ї@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:         Ї@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
№
i
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_695029

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
█
I
-__inference_activation_1_layer_call_fn_694759

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
:         Ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_6907052
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         Ї@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
Ы
б
0__inference_stream_2_conv_1_layer_call_fn_694234

inputs
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_6905282
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Ї: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
∙
n
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_691367

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
:         Ї2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         Ї*
dtype0*
seed╖*
seed2╕2&
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
:         Ї2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ї2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         Ї2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
╖
░
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_689848

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
∙
n
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_694075

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
:         Ї2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         Ї*
dtype0*
seed╖*
seed2╕2&
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
:         Ї2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ї2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         Ї2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
╤
L
0__inference_dense_1_dropout_layer_call_fn_695024

inputs
identity═
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
GPU2*0J 8В *T
fORM
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_6909862
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
№!
╪
C__inference_dense_1_layer_call_and_return_conditional_losses_690825

inputs1
matmul_readvariableop_resource:	└@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2	
BiasAddЙ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const╛
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1╣
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul╣
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add─
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp┤
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2└
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
╫#<2$
"dense_1/kernel/Regularizer/mul_1/x─
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1╕
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityт
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*"
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
╤
L
0__inference_dense_1_dropout_layer_call_fn_695019

inputs
identity═
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
GPU2*0J 8В *T
fORM
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_6907982
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
╢+
ш
O__inference_batch_normalization_layer_call_and_return_conditional_losses_694370

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
щ
U
9__inference_global_average_pooling1d_layer_call_fn_694943

inputs
identity╒
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
GPU2*0J 8В *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_6907672
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
╪
╤
6__inference_batch_normalization_3_layer_call_fn_695108

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6903882
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ў
°
__inference_loss_fn_1_695211T
>stream_1_conv_1_kernel_regularizer_abs_readvariableop_resource:@
identityИв5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpЩ
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
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Absн
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┘
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addў
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>stream_1_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareн
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
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1╪
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1v
IdentityIdentity,stream_1_conv_1/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 2

Identity┴
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
№!
╪
C__inference_dense_1_layer_call_and_return_conditional_losses_695082

inputs1
matmul_readvariableop_resource:	└@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2	
BiasAddЙ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const╛
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1╣
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul╣
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add─
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp┤
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2└
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
╫#<2$
"dense_1/kernel/Regularizer/mul_1/x─
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1╕
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityт
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*"
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
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
╝
r
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_690266

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
ў
°
__inference_loss_fn_2_695231T
>stream_2_conv_1_kernel_regularizer_abs_readvariableop_resource:@
identityИв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЩ
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
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absн
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul┘
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addў
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>stream_2_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Squareн
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
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1╪
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1v
IdentityIdentity,stream_2_conv_1/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 2

Identity┴
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
ч
O
3__inference_stream_2_maxpool_1_layer_call_fn_694836

inputs
identity╘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_6907212
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ·@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
ї
Ц
(__inference_dense_1_layer_call_fn_695057

inputs
unknown:	└@
	unknown_0:@
identityИвStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6908252
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

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
ў
°
__inference_loss_fn_0_695191T
>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource:@
identityИв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpЩ
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
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absн
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul┘
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addў
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Squareн
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
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1╪
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1v
IdentityIdentity,stream_0_conv_1/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 2

Identity┴
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
║
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_694949

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
З+
ъ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_691223

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
:         Ї@2
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
:         Ї@2
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
:         Ї@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         Ї@2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Ї@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
╠
m
4__inference_stream_0_input_drop_layer_call_fn_694031

inputs
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_6913442
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ї2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
Л	
╧
4__inference_batch_normalization_layer_call_fn_694277

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
 :                  @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_6896862
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
Р
m
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_694036

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         Ї2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         Ї2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
М
i
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_694921

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ·@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ·@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
∙,
Н
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_694156

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpy
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
:         Ї2
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
:         Ї@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2	
BiasAddЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const▐
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absн
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul┘
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addф
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Squareн
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
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1╪
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1p
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identity 
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Ї: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
м
O
3__inference_stream_1_maxpool_1_layer_call_fn_694805

inputs
identityх
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'                           * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_6901882
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
╝
Б
*__inference_basemodel_layer_call_fn_693471
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

unknown_17:	└@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИвStatefulPartitionedCall╗
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
:         @*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_6909072
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         Ї:         Ї:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:         Ї
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:         Ї
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:         Ї
"
_user_specified_name
inputs/2
╖
░
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_694656

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
Х
j
N__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_694792

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
-:+                           2

ExpandDims▒
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingVALID*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
∙
n
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_694102

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
:         Ї2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         Ї*
dtype0*
seed╖*
seed2╣2&
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
:         Ї2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ї2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         Ї2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
╕+
ъ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_694530

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
М
i
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_690760

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ·@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ·@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
∙,
Н
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_690600

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpy
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
:         Ї2
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
:         Ї@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2	
BiasAddЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const▐
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absн
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul┘
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addф
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Squareн
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
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1╪
(stream_0_conv_1/kernel/Regularizer/add_1AddV2*stream_0_conv_1/kernel/Regularizer/add:z:0,stream_0_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/add_1p
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identity 
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Ї: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
ч
O
3__inference_stream_1_maxpool_1_layer_call_fn_694810

inputs
identity╘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_6907302
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ·@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
ї
░
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_695128

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
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         @2
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
:         @2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
И
r
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_694977

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
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
я
╤
6__inference_batch_normalization_1_layer_call_fn_694463

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
:         Ї@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6906542
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Ї@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
э
╤
6__inference_batch_normalization_2_layer_call_fn_694636

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
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6912832
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Ї@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
М
i
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_690753

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ·@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ·@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
И
r
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_690774

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
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
ч
O
3__inference_stream_0_maxpool_1_layer_call_fn_694784

inputs
identity╘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_6907392
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ·@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
ї
j
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_691032

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ·@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ·@*
dtype0*
seed╖*
seed2╕2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ·@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ·@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ·@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ·@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
┤
Б
*__inference_basemodel_layer_call_fn_693526
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

unknown_17:	└@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИвStatefulPartitionedCall│
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
:         @*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_6915922
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         Ї:         Ї:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:         Ї
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:         Ї
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:         Ї
"
_user_specified_name
inputs/2
о
j
N__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_694826

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
:         Ї@2

ExpandDimsа
MaxPoolMaxPoolExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2	
MaxPool}
SqueezeSqueezeMaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2	
Squeezei
IdentityIdentitySqueeze:output:0*
T0*,
_output_shapes
:         ·@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
╠
m
4__inference_stream_1_input_drop_layer_call_fn_694058

inputs
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_6913672
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ї2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
╠*
ъ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_695162

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
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

:@*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@2
moments/StopGradientд
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         @2
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
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         @2
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
:         @2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
∙,
Н
K__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_690564

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpв5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpy
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
:         Ї2
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
:         Ї@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2	
BiasAddЩ
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Const▐
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Absн
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┘
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addф
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareн
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
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1╪
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1p
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identity 
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Ї: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
╝
Б
*__inference_basemodel_layer_call_fn_690958
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

unknown_17:	└@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИвStatefulPartitionedCall╗
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
:         @*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_6909072
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         Ї:         Ї:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:         Ї
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:         Ї
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:         Ї
"
_user_specified_name
inputs_2
▐▄
П
E__inference_basemodel_layer_call_and_return_conditional_losses_690907

inputs
inputs_1
inputs_2,
stream_2_conv_1_690529:@$
stream_2_conv_1_690531:@,
stream_1_conv_1_690565:@$
stream_1_conv_1_690567:@,
stream_0_conv_1_690601:@$
stream_0_conv_1_690603:@*
batch_normalization_2_690626:@*
batch_normalization_2_690628:@*
batch_normalization_2_690630:@*
batch_normalization_2_690632:@*
batch_normalization_1_690655:@*
batch_normalization_1_690657:@*
batch_normalization_1_690659:@*
batch_normalization_1_690661:@(
batch_normalization_690684:@(
batch_normalization_690686:@(
batch_normalization_690688:@(
batch_normalization_690690:@!
dense_1_690826:	└@
dense_1_690828:@*
batch_normalization_3_690831:@*
batch_normalization_3_690833:@*
batch_normalization_3_690835:@*
batch_normalization_3_690837:@
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpв'stream_0_conv_1/StatefulPartitionedCallв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpв'stream_1_conv_1/StatefulPartitionedCallв5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв'stream_2_conv_1/StatefulPartitionedCallв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp 
#stream_2_input_drop/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_6904822%
#stream_2_input_drop/PartitionedCall 
#stream_1_input_drop/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_6904892%
#stream_1_input_drop/PartitionedCall¤
#stream_0_input_drop/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_6904962%
#stream_0_input_drop/PartitionedCallх
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_2_input_drop/PartitionedCall:output:0stream_2_conv_1_690529stream_2_conv_1_690531*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_6905282)
'stream_2_conv_1/StatefulPartitionedCallх
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_1_input_drop/PartitionedCall:output:0stream_1_conv_1_690565stream_1_conv_1_690567*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_6905642)
'stream_1_conv_1/StatefulPartitionedCallх
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_690601stream_0_conv_1_690603*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_6906002)
'stream_0_conv_1/StatefulPartitionedCall╟
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_690626batch_normalization_2_690628batch_normalization_2_690630batch_normalization_2_690632*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6906252/
-batch_normalization_2/StatefulPartitionedCall╟
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_690655batch_normalization_1_690657batch_normalization_1_690659batch_normalization_1_690661*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6906542/
-batch_normalization_1/StatefulPartitionedCall╣
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_690684batch_normalization_690686batch_normalization_690688batch_normalization_690690*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_6906832-
+batch_normalization/StatefulPartitionedCallШ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_6906982
activation_2/PartitionedCallШ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_6907052
activation_1/PartitionedCallР
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_6907122
activation/PartitionedCallЩ
"stream_2_maxpool_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_6907212$
"stream_2_maxpool_1/PartitionedCallЩ
"stream_1_maxpool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_6907302$
"stream_1_maxpool_1/PartitionedCallЧ
"stream_0_maxpool_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_6907392$
"stream_0_maxpool_1/PartitionedCallЦ
stream_2_drop_1/PartitionedCallPartitionedCall+stream_2_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_6907462!
stream_2_drop_1/PartitionedCallЦ
stream_1_drop_1/PartitionedCallPartitionedCall+stream_1_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_6907532!
stream_1_drop_1/PartitionedCallЦ
stream_0_drop_1/PartitionedCallPartitionedCall+stream_0_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_6907602!
stream_0_drop_1/PartitionedCallй
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
GPU2*0J 8В *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_6907672*
(global_average_pooling1d/PartitionedCallп
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
GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_6907742,
*global_average_pooling1d_1/PartitionedCallп
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
GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_6907812,
*global_average_pooling1d_2/PartitionedCall°
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
GPU2*0J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_6907912
concatenate/PartitionedCallЛ
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
GPU2*0J 8В *T
fORM
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_6907982!
dense_1_dropout/PartitionedCall┤
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_690826dense_1_690828*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6908252!
dense_1/StatefulPartitionedCall║
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_690831batch_normalization_3_690833batch_normalization_3_690835batch_normalization_3_690837*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6903282/
-batch_normalization_3/StatefulPartitionedCallе
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *W
fRRP
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_6908442$
"dense_activation_1/PartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const╔
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_690601*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absн
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul┘
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/add╧
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_1_690601*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Squareн
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
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1╪
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
(stream_1_conv_1/kernel/Regularizer/Const╔
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_1_conv_1_690565*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Absн
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┘
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/add╧
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_690565*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareн
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
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1╪
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
(stream_2_conv_1/kernel/Regularizer/Const╔
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_690529*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absн
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul┘
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/add╧
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_2_conv_1_690529*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Squareн
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
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1╪
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
 dense_1/kernel/Regularizer/Constо
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_690826*
_output_shapes
:	└@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1╣
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul╣
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add┤
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_690826*
_output_shapes
:	└@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp┤
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2└
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
╫#<2$
"dense_1/kernel/Regularizer/mul_1/x─
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1╕
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Ж
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityш
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_1_conv_1/StatefulPartitionedCall6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         Ї:         Ї:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:         Ї
 
_user_specified_nameinputs:TP
,
_output_shapes
:         Ї
 
_user_specified_nameinputs:TP
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
а
▌
__inference_loss_fn_3_695251I
6dense_1_kernel_regularizer_abs_readvariableop_resource:	└@
identityИв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpЙ
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/Const╓
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	└@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1╣
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul╣
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add▄
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	└@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp┤
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2└
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
╫#<2$
"dense_1/kernel/Regularizer/mul_1/x─
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1╕
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1n
IdentityIdentity$dense_1/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 2

Identity▒
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
╖
░
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_694496

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
Н	
╤
6__inference_batch_normalization_2_layer_call_fn_694610

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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6900702
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
∙,
Н
K__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_690528

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpy
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
:         Ї2
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
:         Ї@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2	
BiasAddЩ
(stream_2_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_2_conv_1/kernel/Regularizer/Const▐
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absн
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul┘
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addф
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Squareн
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
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1╪
(stream_2_conv_1/kernel/Regularizer/add_1AddV2*stream_2_conv_1/kernel/Regularizer/add:z:0,stream_2_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/add_1p
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identity 
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Ї: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
ш
b
F__inference_activation_layer_call_and_return_conditional_losses_690712

inputs
identityS
TanhTanhinputs*
T0*,
_output_shapes
:         Ї@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:         Ї@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
О
░
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_694710

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
:         Ї@2
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
:         Ї@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Ї@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
Э
U
9__inference_global_average_pooling1d_layer_call_fn_694938

inputs
identity▐
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
GPU2*0J 8В *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_6902422
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
с
L
0__inference_stream_0_drop_1_layer_call_fn_694857

inputs
identity╤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_6907602
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ·@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
┌
╤
6__inference_batch_normalization_3_layer_call_fn_695095

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6903282
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
э
W
;__inference_global_average_pooling1d_2_layer_call_fn_694987

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
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_6907812
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
Ж
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_694955

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
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
Юш
Щ
E__inference_basemodel_layer_call_and_return_conditional_losses_691978
inputs_0
inputs_1
inputs_2,
stream_2_conv_1_691846:@$
stream_2_conv_1_691848:@,
stream_1_conv_1_691851:@$
stream_1_conv_1_691853:@,
stream_0_conv_1_691856:@$
stream_0_conv_1_691858:@*
batch_normalization_2_691861:@*
batch_normalization_2_691863:@*
batch_normalization_2_691865:@*
batch_normalization_2_691867:@*
batch_normalization_1_691870:@*
batch_normalization_1_691872:@*
batch_normalization_1_691874:@*
batch_normalization_1_691876:@(
batch_normalization_691879:@(
batch_normalization_691881:@(
batch_normalization_691883:@(
batch_normalization_691885:@!
dense_1_691902:	└@
dense_1_691904:@*
batch_normalization_3_691907:@*
batch_normalization_3_691909:@*
batch_normalization_3_691911:@*
batch_normalization_3_691913:@
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpв'stream_0_conv_1/StatefulPartitionedCallв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpв'stream_0_drop_1/StatefulPartitionedCallв+stream_0_input_drop/StatefulPartitionedCallв'stream_1_conv_1/StatefulPartitionedCallв5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв'stream_1_drop_1/StatefulPartitionedCallв+stream_1_input_drop/StatefulPartitionedCallв'stream_2_conv_1/StatefulPartitionedCallв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpв'stream_2_drop_1/StatefulPartitionedCallв+stream_2_input_drop/StatefulPartitionedCallЧ
+stream_2_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_6913902-
+stream_2_input_drop/StatefulPartitionedCall┼
+stream_1_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_1,^stream_2_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_6913672-
+stream_1_input_drop/StatefulPartitionedCall┼
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_0,^stream_1_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_6913442-
+stream_0_input_drop/StatefulPartitionedCallэ
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_2_input_drop/StatefulPartitionedCall:output:0stream_2_conv_1_691846stream_2_conv_1_691848*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_6905282)
'stream_2_conv_1/StatefulPartitionedCallэ
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_1_input_drop/StatefulPartitionedCall:output:0stream_1_conv_1_691851stream_1_conv_1_691853*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_6905642)
'stream_1_conv_1/StatefulPartitionedCallэ
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_691856stream_0_conv_1_691858*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_6906002)
'stream_0_conv_1/StatefulPartitionedCall┼
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_691861batch_normalization_2_691863batch_normalization_2_691865batch_normalization_2_691867*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6912832/
-batch_normalization_2/StatefulPartitionedCall┼
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_691870batch_normalization_1_691872batch_normalization_1_691874batch_normalization_1_691876*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6912232/
-batch_normalization_1/StatefulPartitionedCall╖
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_691879batch_normalization_691881batch_normalization_691883batch_normalization_691885*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_6911632-
+batch_normalization/StatefulPartitionedCallШ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_6906982
activation_2/PartitionedCallШ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_6907052
activation_1/PartitionedCallР
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_6907122
activation/PartitionedCallЩ
"stream_2_maxpool_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_6907212$
"stream_2_maxpool_1/PartitionedCallЩ
"stream_1_maxpool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_6907302$
"stream_1_maxpool_1/PartitionedCallЧ
"stream_0_maxpool_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_6907392$
"stream_0_maxpool_1/PartitionedCall▄
'stream_2_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_2_maxpool_1/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_6910782)
'stream_2_drop_1/StatefulPartitionedCall╪
'stream_1_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_1_maxpool_1/PartitionedCall:output:0(^stream_2_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_6910552)
'stream_1_drop_1/StatefulPartitionedCall╪
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_0_maxpool_1/PartitionedCall:output:0(^stream_1_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_6910322)
'stream_0_drop_1/StatefulPartitionedCall▒
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
GPU2*0J 8В *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_6907672*
(global_average_pooling1d/PartitionedCall╖
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
GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_6907742,
*global_average_pooling1d_1/PartitionedCall╖
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
GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_6907812,
*global_average_pooling1d_2/PartitionedCall°
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
GPU2*0J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_6907912
concatenate/PartitionedCallЛ
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
GPU2*0J 8В *T
fORM
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_6909862!
dense_1_dropout/PartitionedCall┤
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_691902dense_1_691904*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6908252!
dense_1/StatefulPartitionedCall╕
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_691907batch_normalization_3_691909batch_normalization_3_691911batch_normalization_3_691913*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6903882/
-batch_normalization_3/StatefulPartitionedCallе
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *W
fRRP
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_6908442$
"dense_activation_1/PartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const╔
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_691856*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absн
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul┘
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/add╧
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_1_691856*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Squareн
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
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1╪
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
(stream_1_conv_1/kernel/Regularizer/Const╔
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_1_conv_1_691851*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Absн
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┘
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/add╧
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_691851*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareн
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
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1╪
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
(stream_2_conv_1/kernel/Regularizer/Const╔
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_691846*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absн
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul┘
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/add╧
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_2_conv_1_691846*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Squareн
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
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1╪
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
 dense_1/kernel/Regularizer/Constо
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_691902*
_output_shapes
:	└@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1╣
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul╣
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add┤
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_691902*
_output_shapes
:	└@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp┤
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2└
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
╫#<2$
"dense_1/kernel/Regularizer/mul_1/x─
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1╕
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Ж
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

IdentityЁ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall(^stream_1_conv_1/StatefulPartitionedCall6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_1_drop_1/StatefulPartitionedCall,^stream_1_input_drop/StatefulPartitionedCall(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_2_drop_1/StatefulPartitionedCall,^stream_2_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         Ї:         Ї:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:         Ї
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:         Ї
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:         Ї
"
_user_specified_name
inputs_2
№
i
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_690798

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
э
W
;__inference_global_average_pooling1d_1_layer_call_fn_694965

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
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_6907742
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
╖
░
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_690010

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
╢+
ш
O__inference_batch_normalization_layer_call_and_return_conditional_losses_689746

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
▌z
ф

C__inference_model_1_layer_call_and_return_conditional_losses_692531
left_inputs&
basemodel_692421:@
basemodel_692423:@&
basemodel_692425:@
basemodel_692427:@&
basemodel_692429:@
basemodel_692431:@
basemodel_692433:@
basemodel_692435:@
basemodel_692437:@
basemodel_692439:@
basemodel_692441:@
basemodel_692443:@
basemodel_692445:@
basemodel_692447:@
basemodel_692449:@
basemodel_692451:@
basemodel_692453:@
basemodel_692455:@#
basemodel_692457:	└@
basemodel_692459:@
basemodel_692461:@
basemodel_692463:@
basemodel_692465:@
basemodel_692467:@
identityИв!basemodel/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpв5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpї
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsleft_inputsleft_inputsbasemodel_692421basemodel_692423basemodel_692425basemodel_692427basemodel_692429basemodel_692431basemodel_692433basemodel_692435basemodel_692437basemodel_692439basemodel_692441basemodel_692443basemodel_692445basemodel_692447basemodel_692449basemodel_692451basemodel_692453basemodel_692455basemodel_692457basemodel_692459basemodel_692461basemodel_692463basemodel_692465basemodel_692467*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_6909072#
!basemodel/StatefulPartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const├
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_692429*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absн
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul┘
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/add╔
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_692429*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Squareн
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
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1╪
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
(stream_1_conv_1/kernel/Regularizer/Const├
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_692425*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Absн
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┘
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/add╔
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_692425*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareн
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
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1╪
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
(stream_2_conv_1/kernel/Regularizer/Const├
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_692421*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absн
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul┘
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/add╔
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_692421*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Squareн
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
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1╪
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
 dense_1/kernel/Regularizer/Const░
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_692457*
_output_shapes
:	└@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1╣
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul╣
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add╢
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_692457*
_output_shapes
:	└@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp┤
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2└
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
╫#<2$
"dense_1/kernel/Regularizer/mul_1/x─
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1╕
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Е
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityо
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 2F
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
:         Ї
%
_user_specified_nameleft_inputs
ўб
╪
!__inference__wrapped_model_689662
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
8model_1_basemodel_dense_1_matmul_readvariableop_resource:	└@G
9model_1_basemodel_dense_1_biasadd_readvariableop_resource:@W
Imodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_resource:@[
Mmodel_1_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:@Y
Kmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource:@Y
Kmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource:@
identityИв>model_1/basemodel/batch_normalization/batchnorm/ReadVariableOpв@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1в@model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2вBmodel_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpв@model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpвBmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1вBmodel_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2вDmodel_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpв@model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOpвBmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1вBmodel_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2вDmodel_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpв@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOpвBmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1вBmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2вDmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpв0model_1/basemodel/dense_1/BiasAdd/ReadVariableOpв/model_1/basemodel/dense_1/MatMul/ReadVariableOpв8model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpвDmodel_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpв8model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpвDmodel_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpв8model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpвDmodel_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp░
.model_1/basemodel/stream_2_input_drop/IdentityIdentityleft_inputs*
T0*,
_output_shapes
:         Ї20
.model_1/basemodel/stream_2_input_drop/Identity░
.model_1/basemodel/stream_1_input_drop/IdentityIdentityleft_inputs*
T0*,
_output_shapes
:         Ї20
.model_1/basemodel/stream_1_input_drop/Identity░
.model_1/basemodel/stream_0_input_drop/IdentityIdentityleft_inputs*
T0*,
_output_shapes
:         Ї20
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
:         Ї25
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
:         Ї@*
paddingSAME*
strides
2*
(model_1/basemodel/stream_2_conv_1/conv1d∙
0model_1/basemodel/stream_2_conv_1/conv1d/SqueezeSqueeze1model_1/basemodel/stream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2+
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
:         Ї25
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
:         Ї@*
paddingSAME*
strides
2*
(model_1/basemodel/stream_1_conv_1/conv1d∙
0model_1/basemodel/stream_1_conv_1/conv1d/SqueezeSqueeze1model_1/basemodel/stream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2+
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
:         Ї25
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
:         Ї@*
paddingSAME*
strides
2*
(model_1/basemodel/stream_0_conv_1/conv1d∙
0model_1/basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze1model_1/basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2+
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
:         Ї@29
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
:         Ї@29
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
:         Ї@29
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
:         Ї@29
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
:         Ї@27
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
:         Ї@27
5model_1/basemodel/batch_normalization/batchnorm/add_1╞
#model_1/basemodel/activation_2/TanhTanh;model_1/basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         Ї@2%
#model_1/basemodel/activation_2/Tanh╞
#model_1/basemodel/activation_1/TanhTanh;model_1/basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         Ї@2%
#model_1/basemodel/activation_1/Tanh└
!model_1/basemodel/activation/TanhTanh9model_1/basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         Ї@2#
!model_1/basemodel/activation/Tanhм
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
:         Ї@21
/model_1/basemodel/stream_2_maxpool_1/ExpandDimsП
,model_1/basemodel/stream_2_maxpool_1/MaxPoolMaxPool8model_1/basemodel/stream_2_maxpool_1/ExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2.
,model_1/basemodel/stream_2_maxpool_1/MaxPoolь
,model_1/basemodel/stream_2_maxpool_1/SqueezeSqueeze5model_1/basemodel/stream_2_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2.
,model_1/basemodel/stream_2_maxpool_1/Squeezeм
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
:         Ї@21
/model_1/basemodel/stream_1_maxpool_1/ExpandDimsП
,model_1/basemodel/stream_1_maxpool_1/MaxPoolMaxPool8model_1/basemodel/stream_1_maxpool_1/ExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2.
,model_1/basemodel/stream_1_maxpool_1/MaxPoolь
,model_1/basemodel/stream_1_maxpool_1/SqueezeSqueeze5model_1/basemodel/stream_1_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2.
,model_1/basemodel/stream_1_maxpool_1/Squeezeм
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
:         Ї@21
/model_1/basemodel/stream_0_maxpool_1/ExpandDimsП
,model_1/basemodel/stream_0_maxpool_1/MaxPoolMaxPool8model_1/basemodel/stream_0_maxpool_1/ExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2.
,model_1/basemodel/stream_0_maxpool_1/MaxPoolь
,model_1/basemodel/stream_0_maxpool_1/SqueezeSqueeze5model_1/basemodel/stream_0_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2.
,model_1/basemodel/stream_0_maxpool_1/Squeeze╥
*model_1/basemodel/stream_2_drop_1/IdentityIdentity5model_1/basemodel/stream_2_maxpool_1/Squeeze:output:0*
T0*,
_output_shapes
:         ·@2,
*model_1/basemodel/stream_2_drop_1/Identity╥
*model_1/basemodel/stream_1_drop_1/IdentityIdentity5model_1/basemodel/stream_1_maxpool_1/Squeeze:output:0*
T0*,
_output_shapes
:         ·@2,
*model_1/basemodel/stream_1_drop_1/Identity╥
*model_1/basemodel/stream_0_drop_1/IdentityIdentity5model_1/basemodel/stream_0_maxpool_1/Squeeze:output:0*
T0*,
_output_shapes
:         ·@2,
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
:	└@*
dtype021
/model_1/basemodel/dense_1/MatMul/ReadVariableOpю
 model_1/basemodel/dense_1/MatMulMatMul3model_1/basemodel/dense_1_dropout/Identity:output:07model_1/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2"
 model_1/basemodel/dense_1/MatMul┌
0model_1/basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp9model_1_basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0model_1/basemodel/dense_1/BiasAdd/ReadVariableOpщ
!model_1/basemodel/dense_1/BiasAddBiasAdd*model_1/basemodel/dense_1/MatMul:product:08model_1/basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2#
!model_1/basemodel/dense_1/BiasAddК
@model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpImodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
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
:@27
5model_1/basemodel/batch_normalization_3/batchnorm/add█
7model_1/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt9model_1/basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@29
7model_1/basemodel/batch_normalization_3/batchnorm/RsqrtЦ
Dmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpMmodel_1_basemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpе
5model_1/basemodel/batch_normalization_3/batchnorm/mulMul;model_1/basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Lmodel_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_3/batchnorm/mulТ
7model_1/basemodel/batch_normalization_3/batchnorm/mul_1Mul*model_1/basemodel/dense_1/BiasAdd:output:09model_1/basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @29
7model_1/basemodel/batch_normalization_3/batchnorm/mul_1Р
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpKmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1е
7model_1/basemodel/batch_normalization_3/batchnorm/mul_2MulJmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:value:09model_1/basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@29
7model_1/basemodel/batch_normalization_3/batchnorm/mul_2Р
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpKmodel_1_basemodel_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02D
Bmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2г
5model_1/basemodel/batch_normalization_3/batchnorm/subSubJmodel_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2:value:0;model_1/basemodel/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@27
5model_1/basemodel/batch_normalization_3/batchnorm/subе
7model_1/basemodel/batch_normalization_3/batchnorm/add_1AddV2;model_1/basemodel/batch_normalization_3/batchnorm/mul_1:z:09model_1/basemodel/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @29
7model_1/basemodel/batch_normalization_3/batchnorm/add_1Ц
IdentityIdentity;model_1/basemodel/batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @2

IdentityБ
NoOpNoOp?^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOpA^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_1A^model_1/basemodel/batch_normalization/batchnorm/ReadVariableOp_2C^model_1/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpA^model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOpC^model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1C^model_1/basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2E^model_1/basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp1^model_1/basemodel/dense_1/BiasAdd/ReadVariableOp0^model_1/basemodel/dense_1/MatMul/ReadVariableOp9^model_1/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp9^model_1/basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp9^model_1/basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpE^model_1/basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 2А
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
:         Ї
%
_user_specified_nameleft_inputs
─
i
0__inference_stream_2_drop_1_layer_call_fn_694916

inputs
identityИвStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_6910782
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ·@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
▒╦
ё
E__inference_basemodel_layer_call_and_return_conditional_losses_693725
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
&dense_1_matmul_readvariableop_resource:	└@5
'dense_1_biasadd_readvariableop_resource:@E
7batch_normalization_3_batchnorm_readvariableop_resource:@I
;batch_normalization_3_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_3_batchnorm_readvariableop_1_resource:@G
9batch_normalization_3_batchnorm_readvariableop_2_resource:@
identityИв,batch_normalization/batchnorm/ReadVariableOpв.batch_normalization/batchnorm/ReadVariableOp_1в.batch_normalization/batchnorm/ReadVariableOp_2в0batch_normalization/batchnorm/mul/ReadVariableOpв.batch_normalization_1/batchnorm/ReadVariableOpв0batch_normalization_1/batchnorm/ReadVariableOp_1в0batch_normalization_1/batchnorm/ReadVariableOp_2в2batch_normalization_1/batchnorm/mul/ReadVariableOpв.batch_normalization_2/batchnorm/ReadVariableOpв0batch_normalization_2/batchnorm/ReadVariableOp_1в0batch_normalization_2/batchnorm/ReadVariableOp_2в2batch_normalization_2/batchnorm/mul/ReadVariableOpв.batch_normalization_3/batchnorm/ReadVariableOpв0batch_normalization_3/batchnorm/ReadVariableOp_1в0batch_normalization_3/batchnorm/ReadVariableOp_2в2batch_normalization_3/batchnorm/mul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpв&stream_0_conv_1/BiasAdd/ReadVariableOpв2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpв&stream_1_conv_1/BiasAdd/ReadVariableOpв2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpв5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв&stream_2_conv_1/BiasAdd/ReadVariableOpв2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЙ
stream_2_input_drop/IdentityIdentityinputs_2*
T0*,
_output_shapes
:         Ї2
stream_2_input_drop/IdentityЙ
stream_1_input_drop/IdentityIdentityinputs_1*
T0*,
_output_shapes
:         Ї2
stream_1_input_drop/IdentityЙ
stream_0_input_drop/IdentityIdentityinputs_0*
T0*,
_output_shapes
:         Ї2
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
:         Ї2#
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
:         Ї@*
paddingSAME*
strides
2
stream_2_conv_1/conv1d├
stream_2_conv_1/conv1d/SqueezeSqueezestream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2
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
:         Ї2#
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
:         Ї@*
paddingSAME*
strides
2
stream_1_conv_1/conv1d├
stream_1_conv_1/conv1d/SqueezeSqueezestream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2
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
:         Ї2#
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
:         Ї@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d├
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2
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
:         Ї@2'
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
:         Ї@2'
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
:         Ї@2'
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
:         Ї@2'
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
:         Ї@2%
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
:         Ї@2%
#batch_normalization/batchnorm/add_1Р
activation_2/TanhTanh)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         Ї@2
activation_2/TanhР
activation_1/TanhTanh)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         Ї@2
activation_1/TanhК
activation/TanhTanh'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         Ї@2
activation/TanhИ
!stream_2_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_2_maxpool_1/ExpandDims/dim╩
stream_2_maxpool_1/ExpandDims
ExpandDimsactivation_2/Tanh:y:0*stream_2_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         Ї@2
stream_2_maxpool_1/ExpandDims┘
stream_2_maxpool_1/MaxPoolMaxPool&stream_2_maxpool_1/ExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2
stream_2_maxpool_1/MaxPool╢
stream_2_maxpool_1/SqueezeSqueeze#stream_2_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2
stream_2_maxpool_1/SqueezeИ
!stream_1_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_1_maxpool_1/ExpandDims/dim╩
stream_1_maxpool_1/ExpandDims
ExpandDimsactivation_1/Tanh:y:0*stream_1_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         Ї@2
stream_1_maxpool_1/ExpandDims┘
stream_1_maxpool_1/MaxPoolMaxPool&stream_1_maxpool_1/ExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2
stream_1_maxpool_1/MaxPool╢
stream_1_maxpool_1/SqueezeSqueeze#stream_1_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2
stream_1_maxpool_1/SqueezeИ
!stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_1/ExpandDims/dim╚
stream_0_maxpool_1/ExpandDims
ExpandDimsactivation/Tanh:y:0*stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         Ї@2
stream_0_maxpool_1/ExpandDims┘
stream_0_maxpool_1/MaxPoolMaxPool&stream_0_maxpool_1/ExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_1/MaxPool╢
stream_0_maxpool_1/SqueezeSqueeze#stream_0_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2
stream_0_maxpool_1/SqueezeЬ
stream_2_drop_1/IdentityIdentity#stream_2_maxpool_1/Squeeze:output:0*
T0*,
_output_shapes
:         ·@2
stream_2_drop_1/IdentityЬ
stream_1_drop_1/IdentityIdentity#stream_1_maxpool_1/Squeeze:output:0*
T0*,
_output_shapes
:         ·@2
stream_1_drop_1/IdentityЬ
stream_0_drop_1/IdentityIdentity#stream_0_maxpool_1/Squeeze:output:0*
T0*,
_output_shapes
:         ·@2
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
:	└@*
dtype02
dense_1/MatMul/ReadVariableOpж
dense_1/MatMulMatMul!dense_1_dropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_1/BiasAdd╘
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
%batch_normalization_3/batchnorm/add/yр
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/addе
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/Rsqrtр
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOp▌
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/mul╩
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @2'
%batch_normalization_3/batchnorm/mul_1┌
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1▌
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/mul_2┌
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2█
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/sub▌
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @2'
%batch_normalization_3/batchnorm/add_1Щ
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
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absн
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul┘
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addЇ
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Squareн
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
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1╪
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
(stream_1_conv_1/kernel/Regularizer/Constю
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Absн
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┘
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addЇ
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareн
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
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1╪
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
(stream_2_conv_1/kernel/Regularizer/Constю
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absн
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul┘
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addЇ
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Squareн
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
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1╪
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
 dense_1/kernel/Regularizer/Const╞
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	└@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1╣
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul╣
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add╠
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	└@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp┤
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2└
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
╫#<2$
"dense_1/kernel/Regularizer/mul_1/x─
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1╕
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Д
IdentityIdentity)batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @2

IdentityН
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_1_conv_1/BiasAdd/ReadVariableOp3^stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_2_conv_1/BiasAdd/ReadVariableOp3^stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         Ї:         Ї:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 2\
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
:         Ї
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:         Ї
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:         Ї
"
_user_specified_name
inputs/2
╙
O
3__inference_dense_activation_1_layer_call_fn_695167

inputs
identity╧
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
GPU2*0J 8В *W
fRRP
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_6908442
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
─z
▀

C__inference_model_1_layer_call_and_return_conditional_losses_692095

inputs&
basemodel_691985:@
basemodel_691987:@&
basemodel_691989:@
basemodel_691991:@&
basemodel_691993:@
basemodel_691995:@
basemodel_691997:@
basemodel_691999:@
basemodel_692001:@
basemodel_692003:@
basemodel_692005:@
basemodel_692007:@
basemodel_692009:@
basemodel_692011:@
basemodel_692013:@
basemodel_692015:@
basemodel_692017:@
basemodel_692019:@#
basemodel_692021:	└@
basemodel_692023:@
basemodel_692025:@
basemodel_692027:@
basemodel_692029:@
basemodel_692031:@
identityИв!basemodel/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpв5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpц
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsbasemodel_691985basemodel_691987basemodel_691989basemodel_691991basemodel_691993basemodel_691995basemodel_691997basemodel_691999basemodel_692001basemodel_692003basemodel_692005basemodel_692007basemodel_692009basemodel_692011basemodel_692013basemodel_692015basemodel_692017basemodel_692019basemodel_692021basemodel_692023basemodel_692025basemodel_692027basemodel_692029basemodel_692031*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_6909072#
!basemodel/StatefulPartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const├
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_691993*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absн
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul┘
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/add╔
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_691993*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Squareн
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
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1╪
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
(stream_1_conv_1/kernel/Regularizer/Const├
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_691989*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Absн
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┘
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/add╔
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_691989*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareн
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
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1╪
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
(stream_2_conv_1/kernel/Regularizer/Const├
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_691985*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absн
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul┘
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/add╔
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_691985*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Squareн
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
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1╪
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
 dense_1/kernel/Regularizer/Const░
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_692021*
_output_shapes
:	└@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1╣
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul╣
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add╢
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_692021*
_output_shapes
:	└@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp┤
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2└
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
╫#<2$
"dense_1/kernel/Regularizer/mul_1/x─
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1╕
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Е
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityо
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 2F
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
:         Ї
 
_user_specified_nameinputs
╒z
ф

C__inference_model_1_layer_call_and_return_conditional_losses_692644
left_inputs&
basemodel_692534:@
basemodel_692536:@&
basemodel_692538:@
basemodel_692540:@&
basemodel_692542:@
basemodel_692544:@
basemodel_692546:@
basemodel_692548:@
basemodel_692550:@
basemodel_692552:@
basemodel_692554:@
basemodel_692556:@
basemodel_692558:@
basemodel_692560:@
basemodel_692562:@
basemodel_692564:@
basemodel_692566:@
basemodel_692568:@#
basemodel_692570:	└@
basemodel_692572:@
basemodel_692574:@
basemodel_692576:@
basemodel_692578:@
basemodel_692580:@
identityИв!basemodel/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpв5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpэ
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsleft_inputsleft_inputsbasemodel_692534basemodel_692536basemodel_692538basemodel_692540basemodel_692542basemodel_692544basemodel_692546basemodel_692548basemodel_692550basemodel_692552basemodel_692554basemodel_692556basemodel_692558basemodel_692560basemodel_692562basemodel_692564basemodel_692566basemodel_692568basemodel_692570basemodel_692572basemodel_692574basemodel_692576basemodel_692578basemodel_692580*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_6915922#
!basemodel/StatefulPartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const├
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_692542*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absн
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul┘
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/add╔
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_692542*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Squareн
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
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1╪
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
(stream_1_conv_1/kernel/Regularizer/Const├
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_692538*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Absн
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┘
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/add╔
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_692538*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareн
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
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1╪
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
(stream_2_conv_1/kernel/Regularizer/Const├
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_692534*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absн
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul┘
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/add╔
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_692534*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Squareн
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
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1╪
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
 dense_1/kernel/Regularizer/Const░
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_692570*
_output_shapes
:	└@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1╣
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul╣
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add╢
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_692570*
_output_shapes
:	└@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp┤
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2└
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
╫#<2$
"dense_1/kernel/Regularizer/mul_1/x─
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1╕
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Е
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityо
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 2F
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
:         Ї
%
_user_specified_nameleft_inputs
э
╤
6__inference_batch_normalization_1_layer_call_fn_694476

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
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6912232
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Ї@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
с
L
0__inference_stream_1_drop_1_layer_call_fn_694884

inputs
identity╤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_6907532
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ·@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
Р
m
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_690489

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         Ї2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         Ї2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
▒╦
Х"
C__inference_model_1_layer_call_and_return_conditional_losses_693356

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
0basemodel_dense_1_matmul_readvariableop_resource:	└@?
1basemodel_dense_1_biasadd_readvariableop_resource:@U
Gbasemodel_batch_normalization_3_assignmovingavg_readvariableop_resource:@W
Ibasemodel_batch_normalization_3_assignmovingavg_1_readvariableop_resource:@S
Ebasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:@O
Abasemodel_batch_normalization_3_batchnorm_readvariableop_resource:@
identityИв-basemodel/batch_normalization/AssignMovingAvgв<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOpв/basemodel/batch_normalization/AssignMovingAvg_1в>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOpв6basemodel/batch_normalization/batchnorm/ReadVariableOpв:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpв/basemodel/batch_normalization_1/AssignMovingAvgв>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOpв1basemodel/batch_normalization_1/AssignMovingAvg_1в@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpв8basemodel/batch_normalization_1/batchnorm/ReadVariableOpв<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpв/basemodel/batch_normalization_2/AssignMovingAvgв>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOpв1basemodel/batch_normalization_2/AssignMovingAvg_1в@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpв8basemodel/batch_normalization_2/batchnorm/ReadVariableOpв<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpв/basemodel/batch_normalization_3/AssignMovingAvgв>basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOpв1basemodel/batch_normalization_3/AssignMovingAvg_1в@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpв8basemodel/batch_normalization_3/batchnorm/ReadVariableOpв<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpв(basemodel/dense_1/BiasAdd/ReadVariableOpв'basemodel/dense_1/MatMul/ReadVariableOpв0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpв<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpв0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpв<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpв0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpв<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpв5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЯ
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
:         Ї2+
)basemodel/stream_2_input_drop/dropout/MulР
+basemodel/stream_2_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2-
+basemodel/stream_2_input_drop/dropout/Shapeо
Bbasemodel/stream_2_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_2_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:         Ї*
dtype0*
seed╖*
seed2╣2D
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
:         Ї24
2basemodel/stream_2_input_drop/dropout/GreaterEqual▐
*basemodel/stream_2_input_drop/dropout/CastCast6basemodel/stream_2_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ї2,
*basemodel/stream_2_input_drop/dropout/Castў
+basemodel/stream_2_input_drop/dropout/Mul_1Mul-basemodel/stream_2_input_drop/dropout/Mul:z:0.basemodel/stream_2_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:         Ї2-
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
:         Ї2+
)basemodel/stream_1_input_drop/dropout/MulР
+basemodel/stream_1_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2-
+basemodel/stream_1_input_drop/dropout/Shapeо
Bbasemodel/stream_1_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_1_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:         Ї*
dtype0*
seed╖*
seed2╕2D
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
:         Ї24
2basemodel/stream_1_input_drop/dropout/GreaterEqual▐
*basemodel/stream_1_input_drop/dropout/CastCast6basemodel/stream_1_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ї2,
*basemodel/stream_1_input_drop/dropout/Castў
+basemodel/stream_1_input_drop/dropout/Mul_1Mul-basemodel/stream_1_input_drop/dropout/Mul:z:0.basemodel/stream_1_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:         Ї2-
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
:         Ї2+
)basemodel/stream_0_input_drop/dropout/MulР
+basemodel/stream_0_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2-
+basemodel/stream_0_input_drop/dropout/Shapeо
Bbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:         Ї*
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
:         Ї24
2basemodel/stream_0_input_drop/dropout/GreaterEqual▐
*basemodel/stream_0_input_drop/dropout/CastCast6basemodel/stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ї2,
*basemodel/stream_0_input_drop/dropout/Castў
+basemodel/stream_0_input_drop/dropout/Mul_1Mul-basemodel/stream_0_input_drop/dropout/Mul:z:0.basemodel/stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:         Ї2-
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
:         Ї2-
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
:         Ї@*
paddingSAME*
strides
2"
 basemodel/stream_2_conv_1/conv1dс
(basemodel/stream_2_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2#
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
:         Ї2-
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
:         Ї@*
paddingSAME*
strides
2"
 basemodel/stream_1_conv_1/conv1dс
(basemodel/stream_1_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2#
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
:         Ї2-
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
:         Ї@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1dс
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2#
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
:         Ї@2;
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
:         Ї@21
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
:         Ї@21
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
:         Ї@2;
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
:         Ї@21
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
:         Ї@21
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
:         Ї@29
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
:         Ї@2/
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
:         Ї@2/
-basemodel/batch_normalization/batchnorm/add_1о
basemodel/activation_2/TanhTanh3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         Ї@2
basemodel/activation_2/Tanhо
basemodel/activation_1/TanhTanh3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         Ї@2
basemodel/activation_1/Tanhи
basemodel/activation/TanhTanh1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         Ї@2
basemodel/activation/TanhЬ
+basemodel/stream_2_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_2_maxpool_1/ExpandDims/dimЄ
'basemodel/stream_2_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation_2/Tanh:y:04basemodel/stream_2_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         Ї@2)
'basemodel/stream_2_maxpool_1/ExpandDimsў
$basemodel/stream_2_maxpool_1/MaxPoolMaxPool0basemodel/stream_2_maxpool_1/ExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_2_maxpool_1/MaxPool╘
$basemodel/stream_2_maxpool_1/SqueezeSqueeze-basemodel/stream_2_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2&
$basemodel/stream_2_maxpool_1/SqueezeЬ
+basemodel/stream_1_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_1_maxpool_1/ExpandDims/dimЄ
'basemodel/stream_1_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation_1/Tanh:y:04basemodel/stream_1_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         Ї@2)
'basemodel/stream_1_maxpool_1/ExpandDimsў
$basemodel/stream_1_maxpool_1/MaxPoolMaxPool0basemodel/stream_1_maxpool_1/ExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_1_maxpool_1/MaxPool╘
$basemodel/stream_1_maxpool_1/SqueezeSqueeze-basemodel/stream_1_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2&
$basemodel/stream_1_maxpool_1/SqueezeЬ
+basemodel/stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_0_maxpool_1/ExpandDims/dimЁ
'basemodel/stream_0_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation/Tanh:y:04basemodel/stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         Ї@2)
'basemodel/stream_0_maxpool_1/ExpandDimsў
$basemodel/stream_0_maxpool_1/MaxPoolMaxPool0basemodel/stream_0_maxpool_1/ExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_0_maxpool_1/MaxPool╘
$basemodel/stream_0_maxpool_1/SqueezeSqueeze-basemodel/stream_0_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2&
$basemodel/stream_0_maxpool_1/SqueezeЧ
'basemodel/stream_2_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2)
'basemodel/stream_2_drop_1/dropout/Constэ
%basemodel/stream_2_drop_1/dropout/MulMul-basemodel/stream_2_maxpool_1/Squeeze:output:00basemodel/stream_2_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:         ·@2'
%basemodel/stream_2_drop_1/dropout/Mulп
'basemodel/stream_2_drop_1/dropout/ShapeShape-basemodel/stream_2_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2)
'basemodel/stream_2_drop_1/dropout/Shapeв
>basemodel/stream_2_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_2_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:         ·@*
dtype0*
seed╖*
seed2║2@
>basemodel/stream_2_drop_1/dropout/random_uniform/RandomUniformй
0basemodel/stream_2_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>22
0basemodel/stream_2_drop_1/dropout/GreaterEqual/yл
.basemodel/stream_2_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_2_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_2_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ·@20
.basemodel/stream_2_drop_1/dropout/GreaterEqual╥
&basemodel/stream_2_drop_1/dropout/CastCast2basemodel/stream_2_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ·@2(
&basemodel/stream_2_drop_1/dropout/Castч
'basemodel/stream_2_drop_1/dropout/Mul_1Mul)basemodel/stream_2_drop_1/dropout/Mul:z:0*basemodel/stream_2_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:         ·@2)
'basemodel/stream_2_drop_1/dropout/Mul_1Ч
'basemodel/stream_1_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2)
'basemodel/stream_1_drop_1/dropout/Constэ
%basemodel/stream_1_drop_1/dropout/MulMul-basemodel/stream_1_maxpool_1/Squeeze:output:00basemodel/stream_1_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:         ·@2'
%basemodel/stream_1_drop_1/dropout/Mulп
'basemodel/stream_1_drop_1/dropout/ShapeShape-basemodel/stream_1_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2)
'basemodel/stream_1_drop_1/dropout/Shapeв
>basemodel/stream_1_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_1_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:         ·@*
dtype0*
seed╖*
seed2╣2@
>basemodel/stream_1_drop_1/dropout/random_uniform/RandomUniformй
0basemodel/stream_1_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>22
0basemodel/stream_1_drop_1/dropout/GreaterEqual/yл
.basemodel/stream_1_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_1_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_1_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ·@20
.basemodel/stream_1_drop_1/dropout/GreaterEqual╥
&basemodel/stream_1_drop_1/dropout/CastCast2basemodel/stream_1_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ·@2(
&basemodel/stream_1_drop_1/dropout/Castч
'basemodel/stream_1_drop_1/dropout/Mul_1Mul)basemodel/stream_1_drop_1/dropout/Mul:z:0*basemodel/stream_1_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:         ·@2)
'basemodel/stream_1_drop_1/dropout/Mul_1Ч
'basemodel/stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2)
'basemodel/stream_0_drop_1/dropout/Constэ
%basemodel/stream_0_drop_1/dropout/MulMul-basemodel/stream_0_maxpool_1/Squeeze:output:00basemodel/stream_0_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:         ·@2'
%basemodel/stream_0_drop_1/dropout/Mulп
'basemodel/stream_0_drop_1/dropout/ShapeShape-basemodel/stream_0_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_1/dropout/Shapeв
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:         ·@*
dtype0*
seed╖*
seed2╕2@
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniformй
0basemodel/stream_0_drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>22
0basemodel/stream_0_drop_1/dropout/GreaterEqual/yл
.basemodel/stream_0_drop_1/dropout/GreaterEqualGreaterEqualGbasemodel/stream_0_drop_1/dropout/random_uniform/RandomUniform:output:09basemodel/stream_0_drop_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ·@20
.basemodel/stream_0_drop_1/dropout/GreaterEqual╥
&basemodel/stream_0_drop_1/dropout/CastCast2basemodel/stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ·@2(
&basemodel/stream_0_drop_1/dropout/Castч
'basemodel/stream_0_drop_1/dropout/Mul_1Mul)basemodel/stream_0_drop_1/dropout/Mul:z:0*basemodel/stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:         ·@2)
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
:	└@*
dtype02)
'basemodel/dense_1/MatMul/ReadVariableOp╚
basemodel/dense_1/MatMulMatMul%basemodel/concatenate/concat:output:0/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
basemodel/dense_1/MatMul┬
(basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(basemodel/dense_1/BiasAdd/ReadVariableOp╔
basemodel/dense_1/BiasAddBiasAdd"basemodel/dense_1/MatMul:product:00basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
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

:@*
	keep_dims(2.
,basemodel/batch_normalization_3/moments/mean▄
4basemodel/batch_normalization_3/moments/StopGradientStopGradient5basemodel/batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:@26
4basemodel/batch_normalization_3/moments/StopGradientа
9basemodel/batch_normalization_3/moments/SquaredDifferenceSquaredDifference"basemodel/dense_1/BiasAdd:output:0=basemodel/batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:         @2;
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
:@*
dtype02B
@basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOpа
5basemodel/batch_normalization_3/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@27
5basemodel/batch_normalization_3/AssignMovingAvg_1/subЧ
5basemodel/batch_normalization_3/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_3/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@27
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
:@2/
-basemodel/batch_normalization_3/batchnorm/add├
/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_3/batchnorm/Rsqrt■
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpЕ
-basemodel/batch_normalization_3/batchnorm/mulMul3basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_3/batchnorm/mulЄ
/basemodel/batch_normalization_3/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @21
/basemodel/batch_normalization_3/batchnorm/mul_1√
/basemodel/batch_normalization_3/batchnorm/mul_2Mul8basemodel/batch_normalization_3/moments/Squeeze:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_3/batchnorm/mul_2Є
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
:         @21
/basemodel/batch_normalization_3/batchnorm/add_1Щ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const°
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absн
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul┘
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/add■
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Squareн
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
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1╪
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
(stream_1_conv_1/kernel/Regularizer/Const°
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Absн
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┘
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/add■
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareн
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
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1╪
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
(stream_2_conv_1/kernel/Regularizer/Const°
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absн
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul┘
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/add■
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Squareн
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
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1╪
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
 dense_1/kernel/Regularizer/Const╨
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	└@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1╣
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul╣
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add╓
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	└@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp┤
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2└
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
╫#<2$
"dense_1/kernel/Regularizer/mul_1/x─
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1╕
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1О
IdentityIdentity3basemodel/batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity╣
NoOpNoOp.^basemodel/batch_normalization/AssignMovingAvg=^basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_1?^basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_1/AssignMovingAvg?^basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_1A^basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_2/AssignMovingAvg?^basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_1A^basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp0^basemodel/batch_normalization_3/AssignMovingAvg?^basemodel/batch_normalization_3/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_3/AssignMovingAvg_1A^basemodel/batch_normalization_3/AssignMovingAvg_1/ReadVariableOp9^basemodel/batch_normalization_3/batchnorm/ReadVariableOp=^basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 2^
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
:         Ї
 
_user_specified_nameinputs
Й	
╧
4__inference_batch_normalization_layer_call_fn_694290

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallз
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
GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_6897462
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
ъ
d
H__inference_activation_2_layer_call_and_return_conditional_losses_694774

inputs
identityS
TanhTanhinputs*
T0*,
_output_shapes
:         Ї@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:         Ї@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
Х
j
N__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_690216

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
-:+                           2

ExpandDims▒
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingVALID*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
о
j
N__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_690730

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
:         Ї@2

ExpandDimsа
MaxPoolMaxPoolExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2	
MaxPool}
SqueezeSqueezeMaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2	
Squeezei
IdentityIdentitySqueeze:output:0*
T0*,
_output_shapes
:         ·@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
Ж
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_690767

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
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
Х
j
N__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_694844

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
-:+                           2

ExpandDims▒
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingVALID*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
З+
ъ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_694744

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
:         Ї@2
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
:         Ї@2
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
:         Ї@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         Ї@2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Ї@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
╝
r
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_690290

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
Р
m
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_694063

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         Ї2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         Ї2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
ї
j
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_694879

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ·@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ·@*
dtype0*
seed╖*
seed2╕2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ·@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ·@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ·@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ·@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
М
i
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_694867

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ·@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ·@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
∙i
║
"__inference__traced_restore_695428
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
"assignvariableop_12_dense_1_kernel:	└@.
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
∙
n
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_691390

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
:         Ї2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         Ї*
dtype0*
seed╖*
seed2╣2&
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
:         Ї2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ї2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         Ї2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
З+
ъ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_691283

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
:         Ї@2
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
:         Ї@2
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
:         Ї@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         Ї@2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Ї@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
М
о
O__inference_batch_normalization_layer_call_and_return_conditional_losses_690683

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
:         Ї@2
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
:         Ї@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Ї@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
∙
n
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_694048

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
:         Ї2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         Ї*
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
:         Ї2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ї2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         Ї2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
Н	
╤
6__inference_batch_normalization_1_layer_call_fn_694450

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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6899082
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
ї
j
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_691078

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ·@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ·@*
dtype0*
seed╖*
seed2║2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ·@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ·@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ·@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ·@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
╝z
▀

C__inference_model_1_layer_call_and_return_conditional_losses_692314

inputs&
basemodel_692204:@
basemodel_692206:@&
basemodel_692208:@
basemodel_692210:@&
basemodel_692212:@
basemodel_692214:@
basemodel_692216:@
basemodel_692218:@
basemodel_692220:@
basemodel_692222:@
basemodel_692224:@
basemodel_692226:@
basemodel_692228:@
basemodel_692230:@
basemodel_692232:@
basemodel_692234:@
basemodel_692236:@
basemodel_692238:@#
basemodel_692240:	└@
basemodel_692242:@
basemodel_692244:@
basemodel_692246:@
basemodel_692248:@
basemodel_692250:@
identityИв!basemodel/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpв5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp▐
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsinputsinputsbasemodel_692204basemodel_692206basemodel_692208basemodel_692210basemodel_692212basemodel_692214basemodel_692216basemodel_692218basemodel_692220basemodel_692222basemodel_692224basemodel_692226basemodel_692228basemodel_692230basemodel_692232basemodel_692234basemodel_692236basemodel_692238basemodel_692240basemodel_692242basemodel_692244basemodel_692246basemodel_692248basemodel_692250*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_6915922#
!basemodel/StatefulPartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const├
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_692212*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absн
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul┘
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/add╔
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_692212*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Squareн
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
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1╪
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
(stream_1_conv_1/kernel/Regularizer/Const├
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_692208*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Absн
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┘
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/add╔
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_692208*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareн
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
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1╪
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
(stream_2_conv_1/kernel/Regularizer/Const├
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_692204*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absн
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul┘
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/add╔
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_692204*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Squareн
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
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1╪
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
 dense_1/kernel/Regularizer/Const░
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_692240*
_output_shapes
:	└@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1╣
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul╣
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add╢
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_692240*
_output_shapes
:	└@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp┤
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2└
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
╫#<2$
"dense_1/kernel/Regularizer/mul_1/x─
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1╕
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Е
IdentityIdentity*basemodel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityо
NoOpNoOp"^basemodel/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 2F
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
:         Ї
 
_user_specified_nameinputs
З+
ъ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_694584

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
:         Ї@2
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
:         Ї@2
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
:         Ї@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         Ї@2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Ї@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
Х
j
N__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_694818

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
-:+                           2

ExpandDims▒
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingVALID*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
╦
f
,__inference_concatenate_layer_call_fn_695006
inputs_0
inputs_1
inputs_2
identityс
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
GPU2*0J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_6907912
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
╠
m
4__inference_stream_2_input_drop_layer_call_fn_694085

inputs
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_6913902
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ї2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
Ы
б
0__inference_stream_0_conv_1_layer_call_fn_694126

inputs
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_6906002
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Ї: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
╡
о
O__inference_batch_normalization_layer_call_and_return_conditional_losses_689686

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
П	
╤
6__inference_batch_normalization_2_layer_call_fn_694597

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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6900102
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
ї
j
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_694906

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ·@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ·@*
dtype0*
seed╖*
seed2╣2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ·@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ·@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ·@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ·@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
╕+
ъ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_689908

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
о
j
N__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_694852

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
:         Ї@2

ExpandDimsа
MaxPoolMaxPoolExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2	
MaxPool}
SqueezeSqueezeMaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2	
Squeezei
IdentityIdentitySqueeze:output:0*
T0*,
_output_shapes
:         ·@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
║
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_690242

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
К
g
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_690986

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
∙
n
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_691344

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
:         Ї2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         Ї*
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
:         Ї2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ї2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         Ї2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
с
L
0__inference_stream_2_drop_1_layer_call_fn_694911

inputs
identity╤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_6907462
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ·@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
М
i
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_690746

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ·@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ·@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
О
░
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_694550

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
:         Ї@2
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
:         Ї@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Ї@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
УК
Е
E__inference_basemodel_layer_call_and_return_conditional_losses_694021
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
&dense_1_matmul_readvariableop_resource:	└@5
'dense_1_biasadd_readvariableop_resource:@K
=batch_normalization_3_assignmovingavg_readvariableop_resource:@M
?batch_normalization_3_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_3_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_3_batchnorm_readvariableop_resource:@
identityИв#batch_normalization/AssignMovingAvgв2batch_normalization/AssignMovingAvg/ReadVariableOpв%batch_normalization/AssignMovingAvg_1в4batch_normalization/AssignMovingAvg_1/ReadVariableOpв,batch_normalization/batchnorm/ReadVariableOpв0batch_normalization/batchnorm/mul/ReadVariableOpв%batch_normalization_1/AssignMovingAvgв4batch_normalization_1/AssignMovingAvg/ReadVariableOpв'batch_normalization_1/AssignMovingAvg_1в6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpв.batch_normalization_1/batchnorm/ReadVariableOpв2batch_normalization_1/batchnorm/mul/ReadVariableOpв%batch_normalization_2/AssignMovingAvgв4batch_normalization_2/AssignMovingAvg/ReadVariableOpв'batch_normalization_2/AssignMovingAvg_1в6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpв.batch_normalization_2/batchnorm/ReadVariableOpв2batch_normalization_2/batchnorm/mul/ReadVariableOpв%batch_normalization_3/AssignMovingAvgв4batch_normalization_3/AssignMovingAvg/ReadVariableOpв'batch_normalization_3/AssignMovingAvg_1в6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpв.batch_normalization_3/batchnorm/ReadVariableOpв2batch_normalization_3/batchnorm/mul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpв&stream_0_conv_1/BiasAdd/ReadVariableOpв2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpв&stream_1_conv_1/BiasAdd/ReadVariableOpв2stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpв5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв&stream_2_conv_1/BiasAdd/ReadVariableOpв2stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЛ
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
:         Ї2!
stream_2_input_drop/dropout/Mul~
!stream_2_input_drop/dropout/ShapeShapeinputs_2*
T0*
_output_shapes
:2#
!stream_2_input_drop/dropout/ShapeР
8stream_2_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_2_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:         Ї*
dtype0*
seed╖*
seed2╣2:
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
:         Ї2*
(stream_2_input_drop/dropout/GreaterEqual└
 stream_2_input_drop/dropout/CastCast,stream_2_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ї2"
 stream_2_input_drop/dropout/Cast╧
!stream_2_input_drop/dropout/Mul_1Mul#stream_2_input_drop/dropout/Mul:z:0$stream_2_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:         Ї2#
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
:         Ї2!
stream_1_input_drop/dropout/Mul~
!stream_1_input_drop/dropout/ShapeShapeinputs_1*
T0*
_output_shapes
:2#
!stream_1_input_drop/dropout/ShapeР
8stream_1_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_1_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:         Ї*
dtype0*
seed╖*
seed2╕2:
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
:         Ї2*
(stream_1_input_drop/dropout/GreaterEqual└
 stream_1_input_drop/dropout/CastCast,stream_1_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ї2"
 stream_1_input_drop/dropout/Cast╧
!stream_1_input_drop/dropout/Mul_1Mul#stream_1_input_drop/dropout/Mul:z:0$stream_1_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:         Ї2#
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
:         Ї2!
stream_0_input_drop/dropout/Mul~
!stream_0_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/ShapeР
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:         Ї*
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
:         Ї2*
(stream_0_input_drop/dropout/GreaterEqual└
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ї2"
 stream_0_input_drop/dropout/Cast╧
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:         Ї2#
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
:         Ї2#
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
:         Ї@*
paddingSAME*
strides
2
stream_2_conv_1/conv1d├
stream_2_conv_1/conv1d/SqueezeSqueezestream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2
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
:         Ї2#
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
:         Ї@*
paddingSAME*
strides
2
stream_1_conv_1/conv1d├
stream_1_conv_1/conv1d/SqueezeSqueezestream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2
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
:         Ї2#
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
:         Ї@*
paddingSAME*
strides
2
stream_0_conv_1/conv1d├
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2
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
:         Ї@21
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
:         Ї@2'
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
:         Ї@2'
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
:         Ї@21
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
:         Ї@2'
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
:         Ї@2'
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
:         Ї@2/
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
:         Ї@2%
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
:         Ї@2%
#batch_normalization/batchnorm/add_1Р
activation_2/TanhTanh)batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         Ї@2
activation_2/TanhР
activation_1/TanhTanh)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         Ї@2
activation_1/TanhК
activation/TanhTanh'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         Ї@2
activation/TanhИ
!stream_2_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_2_maxpool_1/ExpandDims/dim╩
stream_2_maxpool_1/ExpandDims
ExpandDimsactivation_2/Tanh:y:0*stream_2_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         Ї@2
stream_2_maxpool_1/ExpandDims┘
stream_2_maxpool_1/MaxPoolMaxPool&stream_2_maxpool_1/ExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2
stream_2_maxpool_1/MaxPool╢
stream_2_maxpool_1/SqueezeSqueeze#stream_2_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2
stream_2_maxpool_1/SqueezeИ
!stream_1_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_1_maxpool_1/ExpandDims/dim╩
stream_1_maxpool_1/ExpandDims
ExpandDimsactivation_1/Tanh:y:0*stream_1_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         Ї@2
stream_1_maxpool_1/ExpandDims┘
stream_1_maxpool_1/MaxPoolMaxPool&stream_1_maxpool_1/ExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2
stream_1_maxpool_1/MaxPool╢
stream_1_maxpool_1/SqueezeSqueeze#stream_1_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2
stream_1_maxpool_1/SqueezeИ
!stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!stream_0_maxpool_1/ExpandDims/dim╚
stream_0_maxpool_1/ExpandDims
ExpandDimsactivation/Tanh:y:0*stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         Ї@2
stream_0_maxpool_1/ExpandDims┘
stream_0_maxpool_1/MaxPoolMaxPool&stream_0_maxpool_1/ExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2
stream_0_maxpool_1/MaxPool╢
stream_0_maxpool_1/SqueezeSqueeze#stream_0_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2
stream_0_maxpool_1/SqueezeГ
stream_2_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
stream_2_drop_1/dropout/Const┼
stream_2_drop_1/dropout/MulMul#stream_2_maxpool_1/Squeeze:output:0&stream_2_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:         ·@2
stream_2_drop_1/dropout/MulС
stream_2_drop_1/dropout/ShapeShape#stream_2_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2
stream_2_drop_1/dropout/ShapeД
4stream_2_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_2_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:         ·@*
dtype0*
seed╖*
seed2║26
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
:         ·@2&
$stream_2_drop_1/dropout/GreaterEqual┤
stream_2_drop_1/dropout/CastCast(stream_2_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ·@2
stream_2_drop_1/dropout/Cast┐
stream_2_drop_1/dropout/Mul_1Mulstream_2_drop_1/dropout/Mul:z:0 stream_2_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:         ·@2
stream_2_drop_1/dropout/Mul_1Г
stream_1_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
stream_1_drop_1/dropout/Const┼
stream_1_drop_1/dropout/MulMul#stream_1_maxpool_1/Squeeze:output:0&stream_1_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:         ·@2
stream_1_drop_1/dropout/MulС
stream_1_drop_1/dropout/ShapeShape#stream_1_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2
stream_1_drop_1/dropout/ShapeД
4stream_1_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_1_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:         ·@*
dtype0*
seed╖*
seed2╣26
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
:         ·@2&
$stream_1_drop_1/dropout/GreaterEqual┤
stream_1_drop_1/dropout/CastCast(stream_1_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ·@2
stream_1_drop_1/dropout/Cast┐
stream_1_drop_1/dropout/Mul_1Mulstream_1_drop_1/dropout/Mul:z:0 stream_1_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:         ·@2
stream_1_drop_1/dropout/Mul_1Г
stream_0_drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
stream_0_drop_1/dropout/Const┼
stream_0_drop_1/dropout/MulMul#stream_0_maxpool_1/Squeeze:output:0&stream_0_drop_1/dropout/Const:output:0*
T0*,
_output_shapes
:         ·@2
stream_0_drop_1/dropout/MulС
stream_0_drop_1/dropout/ShapeShape#stream_0_maxpool_1/Squeeze:output:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/ShapeД
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:         ·@*
dtype0*
seed╖*
seed2╕26
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
:         ·@2&
$stream_0_drop_1/dropout/GreaterEqual┤
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ·@2
stream_0_drop_1/dropout/Cast┐
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:         ·@2
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
:	└@*
dtype02
dense_1/MatMul/ReadVariableOpа
dense_1/MatMulMatMulconcatenate/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
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

:@*
	keep_dims(2$
"batch_normalization_3/moments/mean╛
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

:@2,
*batch_normalization_3/moments/StopGradient°
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:         @21
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

:@*
	keep_dims(2(
&batch_normalization_3/moments/variance┬
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze╩
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
╫#<2-
+batch_normalization_3/AssignMovingAvg/decayц
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOpЁ
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_3/AssignMovingAvg/subч
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2+
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
:@*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp°
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_3/AssignMovingAvg_1/subя
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2-
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
:@2%
#batch_normalization_3/batchnorm/addе
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/Rsqrtр
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOp▌
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/mul╩
%batch_normalization_3/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @2'
%batch_normalization_3/batchnorm/mul_1╙
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_3/batchnorm/mul_2╘
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp┘
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_3/batchnorm/sub▌
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @2'
%batch_normalization_3/batchnorm/add_1Щ
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
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absн
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul┘
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/addЇ
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Squareн
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
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1╪
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
(stream_1_conv_1/kernel/Regularizer/Constю
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Absн
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┘
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addЇ
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareн
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
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1╪
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
(stream_2_conv_1/kernel/Regularizer/Constю
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absн
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul┘
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/addЇ
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Squareн
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
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1╪
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
 dense_1/kernel/Regularizer/Const╞
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	└@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1╣
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul╣
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add╠
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	└@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp┤
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2└
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
╫#<2$
"dense_1/kernel/Regularizer/mul_1/x─
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1╕
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Д
IdentityIdentity)batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity∙
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_1_conv_1/BiasAdd/ReadVariableOp3^stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp'^stream_2_conv_1/BiasAdd/ReadVariableOp3^stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         Ї:         Ї:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 2J
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
:         Ї
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:         Ї
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:         Ї
"
_user_specified_name
inputs/2
ъ
d
H__inference_activation_2_layer_call_and_return_conditional_losses_690698

inputs
identityS
TanhTanhinputs*
T0*,
_output_shapes
:         Ї@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:         Ї@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
я
╤
6__inference_batch_normalization_2_layer_call_fn_694623

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
:         Ї@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6906252
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Ї@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
ш▄
С
E__inference_basemodel_layer_call_and_return_conditional_losses_691838
inputs_0
inputs_1
inputs_2,
stream_2_conv_1_691706:@$
stream_2_conv_1_691708:@,
stream_1_conv_1_691711:@$
stream_1_conv_1_691713:@,
stream_0_conv_1_691716:@$
stream_0_conv_1_691718:@*
batch_normalization_2_691721:@*
batch_normalization_2_691723:@*
batch_normalization_2_691725:@*
batch_normalization_2_691727:@*
batch_normalization_1_691730:@*
batch_normalization_1_691732:@*
batch_normalization_1_691734:@*
batch_normalization_1_691736:@(
batch_normalization_691739:@(
batch_normalization_691741:@(
batch_normalization_691743:@(
batch_normalization_691745:@!
dense_1_691762:	└@
dense_1_691764:@*
batch_normalization_3_691767:@*
batch_normalization_3_691769:@*
batch_normalization_3_691771:@*
batch_normalization_3_691773:@
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpв'stream_0_conv_1/StatefulPartitionedCallв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpв'stream_1_conv_1/StatefulPartitionedCallв5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв'stream_2_conv_1/StatefulPartitionedCallв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp 
#stream_2_input_drop/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_6904822%
#stream_2_input_drop/PartitionedCall 
#stream_1_input_drop/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_6904892%
#stream_1_input_drop/PartitionedCall 
#stream_0_input_drop/PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_6904962%
#stream_0_input_drop/PartitionedCallх
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_2_input_drop/PartitionedCall:output:0stream_2_conv_1_691706stream_2_conv_1_691708*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_6905282)
'stream_2_conv_1/StatefulPartitionedCallх
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_1_input_drop/PartitionedCall:output:0stream_1_conv_1_691711stream_1_conv_1_691713*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_6905642)
'stream_1_conv_1/StatefulPartitionedCallх
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_691716stream_0_conv_1_691718*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_6906002)
'stream_0_conv_1/StatefulPartitionedCall╟
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_691721batch_normalization_2_691723batch_normalization_2_691725batch_normalization_2_691727*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6906252/
-batch_normalization_2/StatefulPartitionedCall╟
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_691730batch_normalization_1_691732batch_normalization_1_691734batch_normalization_1_691736*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6906542/
-batch_normalization_1/StatefulPartitionedCall╣
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_691739batch_normalization_691741batch_normalization_691743batch_normalization_691745*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_6906832-
+batch_normalization/StatefulPartitionedCallШ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_6906982
activation_2/PartitionedCallШ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_6907052
activation_1/PartitionedCallР
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_6907122
activation/PartitionedCallЩ
"stream_2_maxpool_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_6907212$
"stream_2_maxpool_1/PartitionedCallЩ
"stream_1_maxpool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_6907302$
"stream_1_maxpool_1/PartitionedCallЧ
"stream_0_maxpool_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_6907392$
"stream_0_maxpool_1/PartitionedCallЦ
stream_2_drop_1/PartitionedCallPartitionedCall+stream_2_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_6907462!
stream_2_drop_1/PartitionedCallЦ
stream_1_drop_1/PartitionedCallPartitionedCall+stream_1_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_6907532!
stream_1_drop_1/PartitionedCallЦ
stream_0_drop_1/PartitionedCallPartitionedCall+stream_0_maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_6907602!
stream_0_drop_1/PartitionedCallй
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
GPU2*0J 8В *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_6907672*
(global_average_pooling1d/PartitionedCallп
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
GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_6907742,
*global_average_pooling1d_1/PartitionedCallп
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
GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_6907812,
*global_average_pooling1d_2/PartitionedCall°
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
GPU2*0J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_6907912
concatenate/PartitionedCallЛ
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
GPU2*0J 8В *T
fORM
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_6907982!
dense_1_dropout/PartitionedCall┤
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_691762dense_1_691764*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6908252!
dense_1/StatefulPartitionedCall║
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_691767batch_normalization_3_691769batch_normalization_3_691771batch_normalization_3_691773*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6903282/
-batch_normalization_3/StatefulPartitionedCallе
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *W
fRRP
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_6908442$
"dense_activation_1/PartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const╔
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_691716*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absн
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul┘
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/add╧
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_1_691716*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Squareн
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
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1╪
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
(stream_1_conv_1/kernel/Regularizer/Const╔
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_1_conv_1_691711*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Absн
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┘
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/add╧
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_691711*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareн
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
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1╪
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
(stream_2_conv_1/kernel/Regularizer/Const╔
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_691706*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absн
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul┘
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/add╧
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_2_conv_1_691706*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Squareн
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
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1╪
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
 dense_1/kernel/Regularizer/Constо
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_691762*
_output_shapes
:	└@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1╣
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul╣
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add┤
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_691762*
_output_shapes
:	└@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp┤
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2└
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
╫#<2$
"dense_1/kernel/Regularizer/mul_1/x─
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1╕
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Ж
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityш
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_1_conv_1/StatefulPartitionedCall6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         Ї:         Ї:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:         Ї
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:         Ї
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:         Ї
"
_user_specified_name
inputs_2
Фш
Ч
E__inference_basemodel_layer_call_and_return_conditional_losses_691592

inputs
inputs_1
inputs_2,
stream_2_conv_1_691460:@$
stream_2_conv_1_691462:@,
stream_1_conv_1_691465:@$
stream_1_conv_1_691467:@,
stream_0_conv_1_691470:@$
stream_0_conv_1_691472:@*
batch_normalization_2_691475:@*
batch_normalization_2_691477:@*
batch_normalization_2_691479:@*
batch_normalization_2_691481:@*
batch_normalization_1_691484:@*
batch_normalization_1_691486:@*
batch_normalization_1_691488:@*
batch_normalization_1_691490:@(
batch_normalization_691493:@(
batch_normalization_691495:@(
batch_normalization_691497:@(
batch_normalization_691499:@!
dense_1_691516:	└@
dense_1_691518:@*
batch_normalization_3_691521:@*
batch_normalization_3_691523:@*
batch_normalization_3_691525:@*
batch_normalization_3_691527:@
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpв'stream_0_conv_1/StatefulPartitionedCallв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpв'stream_0_drop_1/StatefulPartitionedCallв+stream_0_input_drop/StatefulPartitionedCallв'stream_1_conv_1/StatefulPartitionedCallв5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв'stream_1_drop_1/StatefulPartitionedCallв+stream_1_input_drop/StatefulPartitionedCallв'stream_2_conv_1/StatefulPartitionedCallв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpв'stream_2_drop_1/StatefulPartitionedCallв+stream_2_input_drop/StatefulPartitionedCallЧ
+stream_2_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_6913902-
+stream_2_input_drop/StatefulPartitionedCall┼
+stream_1_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_1,^stream_2_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_6913672-
+stream_1_input_drop/StatefulPartitionedCall├
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs,^stream_1_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_6913442-
+stream_0_input_drop/StatefulPartitionedCallэ
'stream_2_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_2_input_drop/StatefulPartitionedCall:output:0stream_2_conv_1_691460stream_2_conv_1_691462*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_6905282)
'stream_2_conv_1/StatefulPartitionedCallэ
'stream_1_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_1_input_drop/StatefulPartitionedCall:output:0stream_1_conv_1_691465stream_1_conv_1_691467*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_6905642)
'stream_1_conv_1/StatefulPartitionedCallэ
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_691470stream_0_conv_1_691472*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_6906002)
'stream_0_conv_1/StatefulPartitionedCall┼
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall0stream_2_conv_1/StatefulPartitionedCall:output:0batch_normalization_2_691475batch_normalization_2_691477batch_normalization_2_691479batch_normalization_2_691481*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6912832/
-batch_normalization_2/StatefulPartitionedCall┼
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_1_conv_1/StatefulPartitionedCall:output:0batch_normalization_1_691484batch_normalization_1_691486batch_normalization_1_691488batch_normalization_1_691490*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6912232/
-batch_normalization_1/StatefulPartitionedCall╖
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_691493batch_normalization_691495batch_normalization_691497batch_normalization_691499*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_6911632-
+batch_normalization/StatefulPartitionedCallШ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_6906982
activation_2/PartitionedCallШ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_6907052
activation_1/PartitionedCallР
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_6907122
activation/PartitionedCallЩ
"stream_2_maxpool_1/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_6907212$
"stream_2_maxpool_1/PartitionedCallЩ
"stream_1_maxpool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_6907302$
"stream_1_maxpool_1/PartitionedCallЧ
"stream_0_maxpool_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_6907392$
"stream_0_maxpool_1/PartitionedCall▄
'stream_2_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_2_maxpool_1/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_6910782)
'stream_2_drop_1/StatefulPartitionedCall╪
'stream_1_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_1_maxpool_1/PartitionedCall:output:0(^stream_2_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_6910552)
'stream_1_drop_1/StatefulPartitionedCall╪
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall+stream_0_maxpool_1/PartitionedCall:output:0(^stream_1_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_6910322)
'stream_0_drop_1/StatefulPartitionedCall▒
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
GPU2*0J 8В *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_6907672*
(global_average_pooling1d/PartitionedCall╖
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
GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_6907742,
*global_average_pooling1d_1/PartitionedCall╖
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
GPU2*0J 8В *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_6907812,
*global_average_pooling1d_2/PartitionedCall°
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
GPU2*0J 8В *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_6907912
concatenate/PartitionedCallЛ
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
GPU2*0J 8В *T
fORM
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_6909862!
dense_1_dropout/PartitionedCall┤
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_691516dense_1_691518*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6908252!
dense_1/StatefulPartitionedCall╕
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_3_691521batch_normalization_3_691523batch_normalization_3_691525batch_normalization_3_691527*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6903882/
-batch_normalization_3/StatefulPartitionedCallе
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *W
fRRP
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_6908442$
"dense_activation_1/PartitionedCallЩ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const╔
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_691470*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absн
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul┘
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/add╧
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_1_691470*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Squareн
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
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1╪
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
(stream_1_conv_1/kernel/Regularizer/Const╔
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_1_conv_1_691465*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Absн
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┘
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/add╧
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_1_conv_1_691465*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareн
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
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1╪
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
(stream_2_conv_1/kernel/Regularizer/Const╔
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_2_conv_1_691460*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absн
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul┘
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/add╧
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_2_conv_1_691460*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Squareн
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
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1╪
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
 dense_1/kernel/Regularizer/Constо
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_691516*
_output_shapes
:	└@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1╣
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul╣
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add┤
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_691516*
_output_shapes
:	└@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp┤
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2└
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
╫#<2$
"dense_1/kernel/Regularizer/mul_1/x─
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1╕
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1Ж
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

IdentityЁ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall(^stream_1_conv_1/StatefulPartitionedCall6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_1_drop_1/StatefulPartitionedCall,^stream_1_input_drop/StatefulPartitionedCall(^stream_2_conv_1/StatefulPartitionedCall6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp(^stream_2_drop_1/StatefulPartitionedCall,^stream_2_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         Ї:         Ї:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:         Ї
 
_user_specified_nameinputs:TP
,
_output_shapes
:         Ї
 
_user_specified_nameinputs:TP
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
Р
m
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_690482

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         Ї2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         Ї2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
╕+
ъ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_690070

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
О
░
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_690654

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
:         Ї@2
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
:         Ї@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Ї@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
Е+
ш
O__inference_batch_normalization_layer_call_and_return_conditional_losses_694424

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
:         Ї@2
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
:         Ї@2
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
:         Ї@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         Ї@2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Ї@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
М
о
O__inference_batch_normalization_layer_call_and_return_conditional_losses_694390

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
:         Ї@2
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
:         Ї@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Ї@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
╫
G
+__inference_activation_layer_call_fn_694749

inputs
identity╠
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_6907122
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         Ї@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
┤
Б
*__inference_basemodel_layer_call_fn_691698
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

unknown_17:	└@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИвStatefulPartitionedCall│
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
:         @*2
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_basemodel_layer_call_and_return_conditional_losses_6915922
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Л
_input_shapesz
x:         Ї:         Ї:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:         Ї
"
_user_specified_name
inputs_0:VR
,
_output_shapes
:         Ї
"
_user_specified_name
inputs_1:VR
,
_output_shapes
:         Ї
"
_user_specified_name
inputs_2
м
O
3__inference_stream_0_maxpool_1_layer_call_fn_694779

inputs
identityх
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'                           * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_6901602
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
Х
j
N__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_690188

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
-:+                           2

ExpandDims▒
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingVALID*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
─
i
0__inference_stream_1_drop_1_layer_call_fn_694889

inputs
identityИвStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_6910552
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ·@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
∙,
Н
K__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_694210

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpв5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpy
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
:         Ї2
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
:         Ї@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2	
BiasAddЩ
(stream_1_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_1_conv_1/kernel/Regularizer/Const▐
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Absн
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┘
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/addф
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareн
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
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1╪
(stream_1_conv_1/kernel/Regularizer/add_1AddV2*stream_1_conv_1/kernel/Regularizer/add:z:0,stream_1_conv_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/add_1p
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identity 
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Ї: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
о
ц
(__inference_model_1_layer_call_fn_692146
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

unknown_17:	└@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИвStatefulPartitionedCallж
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
:         @*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_6920952
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:         Ї
%
_user_specified_nameleft_inputs
ъ
d
H__inference_activation_1_layer_call_and_return_conditional_losses_690705

inputs
identityS
TanhTanhinputs*
T0*,
_output_shapes
:         Ї@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:         Ї@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
Р
m
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_690496

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         Ї2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         Ї2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
Ы
б
0__inference_stream_1_conv_1_layer_call_fn_694180

inputs
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_6905642
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Ї: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
╕+
ъ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_694690

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
█
I
-__inference_activation_2_layer_call_fn_694769

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
:         Ї@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_6906982
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         Ї@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
ж
ц
(__inference_model_1_layer_call_fn_692418
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

unknown_17:	└@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИвStatefulPartitionedCallЮ
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
:         @*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_6923142
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:         Ї
%
_user_specified_nameleft_inputs
К
g
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_695033

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
╝
r
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_694971

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
щ
P
4__inference_stream_1_input_drop_layer_call_fn_694053

inputs
identity╒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_6904892
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
╝
r
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_694993

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
▄

G__inference_concatenate_layer_call_and_return_conditional_losses_690791

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
ї
j
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_691055

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ·@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ·@*
dtype0*
seed╖*
seed2╣2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ·@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ·@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ·@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ·@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
ч
Б
G__inference_concatenate_layer_call_and_return_conditional_losses_695014
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
м
O
3__inference_stream_2_maxpool_1_layer_call_fn_694831

inputs
identityх
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'                           * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_6902162
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
К
j
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_695171

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ї
j
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_694933

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ·@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ·@*
dtype0*
seed╖*
seed2║2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ·@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ·@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ·@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ·@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
о
j
N__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_694800

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
:         Ї@2

ExpandDimsа
MaxPoolMaxPoolExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2	
MaxPool}
SqueezeSqueezeMaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2	
Squeezei
IdentityIdentitySqueeze:output:0*
T0*,
_output_shapes
:         ·@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
ы
╧
4__inference_batch_normalization_layer_call_fn_694303

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
:         Ї@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_6906832
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Ї@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
б
W
;__inference_global_average_pooling1d_1_layer_call_fn_694960

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
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_6902662
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
П	
╤
6__inference_batch_normalization_1_layer_call_fn_694437

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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6898482
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
╠*
ъ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_690388

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
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

:@*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@2
moments/StopGradientд
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         @2
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
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         @2
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
:         @2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
─
i
0__inference_stream_0_drop_1_layer_call_fn_694862

inputs
identityИвStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ·@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_6910322
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ·@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
Ч
с
(__inference_model_1_layer_call_fn_692865

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

unknown_17:	└@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИвStatefulPartitionedCallЩ
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
:         @*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_6923142
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
Х
j
N__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_690160

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
-:+                           2

ExpandDims▒
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+                           *
ksize
*
paddingVALID*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'                           *
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
М
i
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_694894

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ·@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ·@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
Зя
▒
C__inference_model_1_layer_call_and_return_conditional_losses_693062

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
0basemodel_dense_1_matmul_readvariableop_resource:	└@?
1basemodel_dense_1_biasadd_readvariableop_resource:@O
Abasemodel_batch_normalization_3_batchnorm_readvariableop_resource:@S
Ebasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:@Q
Cbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource:@Q
Cbasemodel_batch_normalization_3_batchnorm_readvariableop_2_resource:@
identityИв6basemodel/batch_normalization/batchnorm/ReadVariableOpв8basemodel/batch_normalization/batchnorm/ReadVariableOp_1в8basemodel/batch_normalization/batchnorm/ReadVariableOp_2в:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpв8basemodel/batch_normalization_1/batchnorm/ReadVariableOpв:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1в:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2в<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpв8basemodel/batch_normalization_2/batchnorm/ReadVariableOpв:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1в:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2в<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpв8basemodel/batch_normalization_3/batchnorm/ReadVariableOpв:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1в:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2в<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpв(basemodel/dense_1/BiasAdd/ReadVariableOpв'basemodel/dense_1/MatMul/ReadVariableOpв0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpв<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpв0basemodel/stream_1_conv_1/BiasAdd/ReadVariableOpв<basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOpв0basemodel/stream_2_conv_1/BiasAdd/ReadVariableOpв<basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpв5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpв5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpЫ
&basemodel/stream_2_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:         Ї2(
&basemodel/stream_2_input_drop/IdentityЫ
&basemodel/stream_1_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:         Ї2(
&basemodel/stream_1_input_drop/IdentityЫ
&basemodel/stream_0_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:         Ї2(
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
:         Ї2-
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
:         Ї@*
paddingSAME*
strides
2"
 basemodel/stream_2_conv_1/conv1dс
(basemodel/stream_2_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_2_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2#
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
:         Ї2-
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
:         Ї@*
paddingSAME*
strides
2"
 basemodel/stream_1_conv_1/conv1dс
(basemodel/stream_1_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_1_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2#
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
:         Ї2-
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
:         Ї@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1dс
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
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
:         Ї@2#
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
:         Ї@21
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
:         Ї@21
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
:         Ї@21
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
:         Ї@21
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
:         Ї@2/
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
:         Ї@2/
-basemodel/batch_normalization/batchnorm/add_1о
basemodel/activation_2/TanhTanh3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         Ї@2
basemodel/activation_2/Tanhо
basemodel/activation_1/TanhTanh3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         Ї@2
basemodel/activation_1/Tanhи
basemodel/activation/TanhTanh1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         Ї@2
basemodel/activation/TanhЬ
+basemodel/stream_2_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_2_maxpool_1/ExpandDims/dimЄ
'basemodel/stream_2_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation_2/Tanh:y:04basemodel/stream_2_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         Ї@2)
'basemodel/stream_2_maxpool_1/ExpandDimsў
$basemodel/stream_2_maxpool_1/MaxPoolMaxPool0basemodel/stream_2_maxpool_1/ExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_2_maxpool_1/MaxPool╘
$basemodel/stream_2_maxpool_1/SqueezeSqueeze-basemodel/stream_2_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2&
$basemodel/stream_2_maxpool_1/SqueezeЬ
+basemodel/stream_1_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_1_maxpool_1/ExpandDims/dimЄ
'basemodel/stream_1_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation_1/Tanh:y:04basemodel/stream_1_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         Ї@2)
'basemodel/stream_1_maxpool_1/ExpandDimsў
$basemodel/stream_1_maxpool_1/MaxPoolMaxPool0basemodel/stream_1_maxpool_1/ExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_1_maxpool_1/MaxPool╘
$basemodel/stream_1_maxpool_1/SqueezeSqueeze-basemodel/stream_1_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2&
$basemodel/stream_1_maxpool_1/SqueezeЬ
+basemodel/stream_0_maxpool_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+basemodel/stream_0_maxpool_1/ExpandDims/dimЁ
'basemodel/stream_0_maxpool_1/ExpandDims
ExpandDimsbasemodel/activation/Tanh:y:04basemodel/stream_0_maxpool_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         Ї@2)
'basemodel/stream_0_maxpool_1/ExpandDimsў
$basemodel/stream_0_maxpool_1/MaxPoolMaxPool0basemodel/stream_0_maxpool_1/ExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2&
$basemodel/stream_0_maxpool_1/MaxPool╘
$basemodel/stream_0_maxpool_1/SqueezeSqueeze-basemodel/stream_0_maxpool_1/MaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2&
$basemodel/stream_0_maxpool_1/Squeeze║
"basemodel/stream_2_drop_1/IdentityIdentity-basemodel/stream_2_maxpool_1/Squeeze:output:0*
T0*,
_output_shapes
:         ·@2$
"basemodel/stream_2_drop_1/Identity║
"basemodel/stream_1_drop_1/IdentityIdentity-basemodel/stream_1_maxpool_1/Squeeze:output:0*
T0*,
_output_shapes
:         ·@2$
"basemodel/stream_1_drop_1/Identity║
"basemodel/stream_0_drop_1/IdentityIdentity-basemodel/stream_0_maxpool_1/Squeeze:output:0*
T0*,
_output_shapes
:         ·@2$
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
:	└@*
dtype02)
'basemodel/dense_1/MatMul/ReadVariableOp╬
basemodel/dense_1/MatMulMatMul+basemodel/dense_1_dropout/Identity:output:0/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
basemodel/dense_1/MatMul┬
(basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(basemodel/dense_1/BiasAdd/ReadVariableOp╔
basemodel/dense_1/BiasAddBiasAdd"basemodel/dense_1/MatMul:product:00basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
basemodel/dense_1/BiasAddЄ
8basemodel/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
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
:@2/
-basemodel/batch_normalization_3/batchnorm/add├
/basemodel/batch_normalization_3/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_3/batchnorm/Rsqrt■
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOpЕ
-basemodel/batch_normalization_3/batchnorm/mulMul3basemodel/batch_normalization_3/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2/
-basemodel/batch_normalization_3/batchnorm/mulЄ
/basemodel/batch_normalization_3/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @21
/basemodel/batch_normalization_3/batchnorm/mul_1°
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1Е
/basemodel/batch_normalization_3/batchnorm/mul_2MulBbasemodel/batch_normalization_3/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_3/batchnorm/mul_2°
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
:         @21
/basemodel/batch_normalization_3/batchnorm/add_1Щ
(stream_0_conv_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(stream_0_conv_1/kernel/Regularizer/Const°
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_0_conv_1/kernel/Regularizer/Absн
*stream_0_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_0_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_0_conv_1/kernel/Regularizer/mul/x▄
&stream_0_conv_1/kernel/Regularizer/mulMul1stream_0_conv_1/kernel/Regularizer/mul/x:output:0/stream_0_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/mul┘
&stream_0_conv_1/kernel/Regularizer/addAddV21stream_0_conv_1/kernel/Regularizer/Const:output:0*stream_0_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_0_conv_1/kernel/Regularizer/add■
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_1/kernel/Regularizer/SquareSquare@stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_0_conv_1/kernel/Regularizer/Squareн
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
(stream_0_conv_1/kernel/Regularizer/Sum_1Э
*stream_0_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_0_conv_1/kernel/Regularizer/mul_1/xф
(stream_0_conv_1/kernel/Regularizer/mul_1Mul3stream_0_conv_1/kernel/Regularizer/mul_1/x:output:01stream_0_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_0_conv_1/kernel/Regularizer/mul_1╪
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
(stream_1_conv_1/kernel/Regularizer/Const°
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_1_conv_1/kernel/Regularizer/AbsAbs=stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_1_conv_1/kernel/Regularizer/Absн
*stream_1_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_1_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_1_conv_1/kernel/Regularizer/mul/x▄
&stream_1_conv_1/kernel/Regularizer/mulMul1stream_1_conv_1/kernel/Regularizer/mul/x:output:0/stream_1_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/mul┘
&stream_1_conv_1/kernel/Regularizer/addAddV21stream_1_conv_1/kernel/Regularizer/Const:output:0*stream_1_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_1_conv_1/kernel/Regularizer/add■
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_1_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_1_conv_1/kernel/Regularizer/SquareSquare@stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_1_conv_1/kernel/Regularizer/Squareн
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
(stream_1_conv_1/kernel/Regularizer/Sum_1Э
*stream_1_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_1_conv_1/kernel/Regularizer/mul_1/xф
(stream_1_conv_1/kernel/Regularizer/mul_1Mul3stream_1_conv_1/kernel/Regularizer/mul_1/x:output:01stream_1_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_1_conv_1/kernel/Regularizer/mul_1╪
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
(stream_2_conv_1/kernel/Regularizer/Const°
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype027
5stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_2_conv_1/kernel/Regularizer/AbsAbs=stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2(
&stream_2_conv_1/kernel/Regularizer/Absн
*stream_2_conv_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*stream_2_conv_1/kernel/Regularizer/Const_1┘
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
╫#<2*
(stream_2_conv_1/kernel/Regularizer/mul/x▄
&stream_2_conv_1/kernel/Regularizer/mulMul1stream_2_conv_1/kernel/Regularizer/mul/x:output:0/stream_2_conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/mul┘
&stream_2_conv_1/kernel/Regularizer/addAddV21stream_2_conv_1/kernel/Regularizer/Const:output:0*stream_2_conv_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&stream_2_conv_1/kernel/Regularizer/add■
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_2_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02:
8stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp╧
)stream_2_conv_1/kernel/Regularizer/SquareSquare@stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2+
)stream_2_conv_1/kernel/Regularizer/Squareн
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
(stream_2_conv_1/kernel/Regularizer/Sum_1Э
*stream_2_conv_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2,
*stream_2_conv_1/kernel/Regularizer/mul_1/xф
(stream_2_conv_1/kernel/Regularizer/mul_1Mul3stream_2_conv_1/kernel/Regularizer/mul_1/x:output:01stream_2_conv_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2*
(stream_2_conv_1/kernel/Regularizer/mul_1╪
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
 dense_1/kernel/Regularizer/Const╨
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	└@*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpи
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2 
dense_1/kernel/Regularizer/AbsЩ
"dense_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_1╣
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
╫#<2"
 dense_1/kernel/Regularizer/mul/x╝
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul╣
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/Const:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add╓
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes
:	└@*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp┤
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	└@2#
!dense_1/kernel/Regularizer/SquareЩ
"dense_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/kernel/Regularizer/Const_2└
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
╫#<2$
"dense_1/kernel/Regularizer/mul_1/x─
 dense_1/kernel/Regularizer/mul_1Mul+dense_1/kernel/Regularizer/mul_1/x:output:0)dense_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/mul_1╕
 dense_1/kernel/Regularizer/add_1AddV2"dense_1/kernel/Regularizer/add:z:0$dense_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_1/kernel/Regularizer/add_1О
IdentityIdentity3basemodel/batch_normalization_3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity¤
NoOpNoOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp9^basemodel/batch_normalization/batchnorm/ReadVariableOp_19^basemodel/batch_normalization/batchnorm/ReadVariableOp_2;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization_3/batchnorm/ReadVariableOp;^basemodel/batch_normalization_3/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_3/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_3/batchnorm/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_1_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_1_conv_1/conv1d/ExpandDims_1/ReadVariableOp1^basemodel/stream_2_conv_1/BiasAdd/ReadVariableOp=^basemodel/stream_2_conv_1/conv1d/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_1_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_1_conv_1/kernel/Regularizer/Square/ReadVariableOp6^stream_2_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_2_conv_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 2p
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
:         Ї
 
_user_specified_nameinputs
о
j
N__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_690721

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
:         Ї@2

ExpandDimsа
MaxPoolMaxPoolExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2	
MaxPool}
SqueezeSqueezeMaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2	
Squeezei
IdentityIdentitySqueeze:output:0*
T0*,
_output_shapes
:         ·@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
о
j
N__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_690739

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
:         Ї@2

ExpandDimsа
MaxPoolMaxPoolExpandDims:output:0*0
_output_shapes
:         ·@*
ksize
*
paddingVALID*
strides
2	
MaxPool}
SqueezeSqueezeMaxPool:output:0*
T0*,
_output_shapes
:         ·@*
squeeze_dims
2	
Squeezei
IdentityIdentitySqueeze:output:0*
T0*,
_output_shapes
:         ·@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
щ
P
4__inference_stream_2_input_drop_layer_call_fn_694080

inputs
identity╒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_6904822
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
Р
m
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_694090

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         Ї2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         Ї2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
╡
о
O__inference_batch_normalization_layer_call_and_return_conditional_losses_694336

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
И
r
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_690781

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
:         ·@:T P
,
_output_shapes
:         ·@
 
_user_specified_nameinputs
щ
P
4__inference_stream_0_input_drop_layer_call_fn_694026

inputs
identity╒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_6904962
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ї:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
б
W
;__inference_global_average_pooling1d_2_layer_call_fn_694982

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
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_6902902
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
О
░
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_690625

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
:         Ї@2
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
:         Ї@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         Ї@2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Ї@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
Е+
ш
O__inference_batch_normalization_layer_call_and_return_conditional_losses_691163

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
:         Ї@2
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
:         Ї@2
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
:         Ї@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         Ї@2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Ї@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
ї
░
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_690328

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
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         @2
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
:         @2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Я
с
(__inference_model_1_layer_call_fn_692812

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

unknown_17:	└@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityИвStatefulPartitionedCallб
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
:         @*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_6920952
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:         Ї: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ї
 
_user_specified_nameinputs
К
j
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_690844

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
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
serving_default_left_inputs:0         Ї=
	basemodel0
StatefulPartitionedCall:0         @tensorflow/serving/predict:ь№
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
╣__call__
+║&call_and_return_all_conditional_losses
╗_default_save_signature"
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
╝__call__
+╜&call_and_return_all_conditional_losses"
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
╓
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
╬

Alayers
trainable_variables
Blayer_regularization_losses
Cmetrics
	variables
regularization_losses
Dnon_trainable_variables
Elayer_metrics
╣__call__
╗_default_save_signature
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
-
╛serving_default"
signature_map
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
з
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
┐__call__
+└&call_and_return_all_conditional_losses"
_tf_keras_layer
з
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
┴__call__
+┬&call_and_return_all_conditional_losses"
_tf_keras_layer
з
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
├__call__
+─&call_and_return_all_conditional_losses"
_tf_keras_layer
╜

)kernel
*bias
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
┼__call__
+╞&call_and_return_all_conditional_losses"
_tf_keras_layer
╜

+kernel
,bias
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
╟__call__
+╚&call_and_return_all_conditional_losses"
_tf_keras_layer
╜

-kernel
.bias
Ztrainable_variables
[	variables
\regularization_losses
]	keras_api
╔__call__
+╩&call_and_return_all_conditional_losses"
_tf_keras_layer
ь
^axis
	/gamma
0beta
9moving_mean
:moving_variance
_trainable_variables
`	variables
aregularization_losses
b	keras_api
╦__call__
+╠&call_and_return_all_conditional_losses"
_tf_keras_layer
ь
caxis
	1gamma
2beta
;moving_mean
<moving_variance
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
═__call__
+╬&call_and_return_all_conditional_losses"
_tf_keras_layer
ь
haxis
	3gamma
4beta
=moving_mean
>moving_variance
itrainable_variables
j	variables
kregularization_losses
l	keras_api
╧__call__
+╨&call_and_return_all_conditional_losses"
_tf_keras_layer
з
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
╤__call__
+╥&call_and_return_all_conditional_losses"
_tf_keras_layer
з
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
╙__call__
+╘&call_and_return_all_conditional_losses"
_tf_keras_layer
з
utrainable_variables
v	variables
wregularization_losses
x	keras_api
╒__call__
+╓&call_and_return_all_conditional_losses"
_tf_keras_layer
з
ytrainable_variables
z	variables
{regularization_losses
|	keras_api
╫__call__
+╪&call_and_return_all_conditional_losses"
_tf_keras_layer
и
}trainable_variables
~	variables
regularization_losses
А	keras_api
┘__call__
+┌&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Бtrainable_variables
В	variables
Гregularization_losses
Д	keras_api
█__call__
+▄&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Еtrainable_variables
Ж	variables
Зregularization_losses
И	keras_api
▌__call__
+▐&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Йtrainable_variables
К	variables
Лregularization_losses
М	keras_api
▀__call__
+р&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Нtrainable_variables
О	variables
Пregularization_losses
Р	keras_api
с__call__
+т&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Сtrainable_variables
Т	variables
Уregularization_losses
Ф	keras_api
у__call__
+ф&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Хtrainable_variables
Ц	variables
Чregularization_losses
Ш	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Щtrainable_variables
Ъ	variables
Ыregularization_losses
Ь	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Эtrainable_variables
Ю	variables
Яregularization_losses
а	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses"
_tf_keras_layer
л
бtrainable_variables
в	variables
гregularization_losses
д	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses"
_tf_keras_layer
┴

5kernel
6bias
еtrainable_variables
ж	variables
зregularization_losses
и	keras_api
э__call__
+ю&call_and_return_all_conditional_losses"
_tf_keras_layer
ё
	йaxis
	7gamma
8beta
?moving_mean
@moving_variance
кtrainable_variables
л	variables
мregularization_losses
н	keras_api
я__call__
+Ё&call_and_return_all_conditional_losses"
_tf_keras_layer
л
оtrainable_variables
п	variables
░regularization_losses
▒	keras_api
ё__call__
+Є&call_and_return_all_conditional_losses"
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
╓
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
є0
Ї1
ї2
Ў3"
trackable_list_wrapper
╡
▓layers
%trainable_variables
 │layer_regularization_losses
┤metrics
&	variables
'regularization_losses
╡non_trainable_variables
╢layer_metrics
╝__call__
+╜&call_and_return_all_conditional_losses
'╜"call_and_return_conditional_losses"
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
!:	└@2dense_1/kernel
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
╡
╖layers
Ftrainable_variables
 ╕layer_regularization_losses
╣metrics
G	variables
Hregularization_losses
║non_trainable_variables
╗layer_metrics
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
╡
╝layers
Jtrainable_variables
 ╜layer_regularization_losses
╛metrics
K	variables
Lregularization_losses
┐non_trainable_variables
└layer_metrics
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
╡
┴layers
Ntrainable_variables
 ┬layer_regularization_losses
├metrics
O	variables
Pregularization_losses
─non_trainable_variables
┼layer_metrics
├__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
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
є0"
trackable_list_wrapper
╡
╞layers
Rtrainable_variables
 ╟layer_regularization_losses
╚metrics
S	variables
Tregularization_losses
╔non_trainable_variables
╩layer_metrics
┼__call__
+╞&call_and_return_all_conditional_losses
'╞"call_and_return_conditional_losses"
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
Ї0"
trackable_list_wrapper
╡
╦layers
Vtrainable_variables
 ╠layer_regularization_losses
═metrics
W	variables
Xregularization_losses
╬non_trainable_variables
╧layer_metrics
╟__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
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
ї0"
trackable_list_wrapper
╡
╨layers
Ztrainable_variables
 ╤layer_regularization_losses
╥metrics
[	variables
\regularization_losses
╙non_trainable_variables
╘layer_metrics
╔__call__
+╩&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses"
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
╡
╒layers
_trainable_variables
 ╓layer_regularization_losses
╫metrics
`	variables
aregularization_losses
╪non_trainable_variables
┘layer_metrics
╦__call__
+╠&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses"
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
╡
┌layers
dtrainable_variables
 █layer_regularization_losses
▄metrics
e	variables
fregularization_losses
▌non_trainable_variables
▐layer_metrics
═__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses"
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
╡
▀layers
itrainable_variables
 рlayer_regularization_losses
сmetrics
j	variables
kregularization_losses
тnon_trainable_variables
уlayer_metrics
╧__call__
+╨&call_and_return_all_conditional_losses
'╨"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
фlayers
mtrainable_variables
 хlayer_regularization_losses
цmetrics
n	variables
oregularization_losses
чnon_trainable_variables
шlayer_metrics
╤__call__
+╥&call_and_return_all_conditional_losses
'╥"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
щlayers
qtrainable_variables
 ъlayer_regularization_losses
ыmetrics
r	variables
sregularization_losses
ьnon_trainable_variables
эlayer_metrics
╙__call__
+╘&call_and_return_all_conditional_losses
'╘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
юlayers
utrainable_variables
 яlayer_regularization_losses
Ёmetrics
v	variables
wregularization_losses
ёnon_trainable_variables
Єlayer_metrics
╒__call__
+╓&call_and_return_all_conditional_losses
'╓"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
єlayers
ytrainable_variables
 Їlayer_regularization_losses
їmetrics
z	variables
{regularization_losses
Ўnon_trainable_variables
ўlayer_metrics
╫__call__
+╪&call_and_return_all_conditional_losses
'╪"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
°layers
}trainable_variables
 ∙layer_regularization_losses
·metrics
~	variables
regularization_losses
√non_trainable_variables
№layer_metrics
┘__call__
+┌&call_and_return_all_conditional_losses
'┌"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
¤layers
Бtrainable_variables
 ■layer_regularization_losses
 metrics
В	variables
Гregularization_losses
Аnon_trainable_variables
Бlayer_metrics
█__call__
+▄&call_and_return_all_conditional_losses
'▄"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Вlayers
Еtrainable_variables
 Гlayer_regularization_losses
Дmetrics
Ж	variables
Зregularization_losses
Еnon_trainable_variables
Жlayer_metrics
▌__call__
+▐&call_and_return_all_conditional_losses
'▐"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Зlayers
Йtrainable_variables
 Иlayer_regularization_losses
Йmetrics
К	variables
Лregularization_losses
Кnon_trainable_variables
Лlayer_metrics
▀__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Мlayers
Нtrainable_variables
 Нlayer_regularization_losses
Оmetrics
О	variables
Пregularization_losses
Пnon_trainable_variables
Рlayer_metrics
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Сlayers
Сtrainable_variables
 Тlayer_regularization_losses
Уmetrics
Т	variables
Уregularization_losses
Фnon_trainable_variables
Хlayer_metrics
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Цlayers
Хtrainable_variables
 Чlayer_regularization_losses
Шmetrics
Ц	variables
Чregularization_losses
Щnon_trainable_variables
Ъlayer_metrics
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Ыlayers
Щtrainable_variables
 Ьlayer_regularization_losses
Эmetrics
Ъ	variables
Ыregularization_losses
Юnon_trainable_variables
Яlayer_metrics
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
аlayers
Эtrainable_variables
 бlayer_regularization_losses
вmetrics
Ю	variables
Яregularization_losses
гnon_trainable_variables
дlayer_metrics
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
еlayers
бtrainable_variables
 жlayer_regularization_losses
зmetrics
в	variables
гregularization_losses
иnon_trainable_variables
йlayer_metrics
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
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
Ў0"
trackable_list_wrapper
╕
кlayers
еtrainable_variables
 лlayer_regularization_losses
мmetrics
ж	variables
зregularization_losses
нnon_trainable_variables
оlayer_metrics
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
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
╕
пlayers
кtrainable_variables
 ░layer_regularization_losses
▒metrics
л	variables
мregularization_losses
▓non_trainable_variables
│layer_metrics
я__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┤layers
оtrainable_variables
 ╡layer_regularization_losses
╢metrics
п	variables
░regularization_losses
╖non_trainable_variables
╕layer_metrics
ё__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
■
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
є0"
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
Ї0"
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
ї0"
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
Ў0"
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
ю2ы
(__inference_model_1_layer_call_fn_692146
(__inference_model_1_layer_call_fn_692812
(__inference_model_1_layer_call_fn_692865
(__inference_model_1_layer_call_fn_692418└
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
┌2╫
C__inference_model_1_layer_call_and_return_conditional_losses_693062
C__inference_model_1_layer_call_and_return_conditional_losses_693356
C__inference_model_1_layer_call_and_return_conditional_losses_692531
C__inference_model_1_layer_call_and_return_conditional_losses_692644└
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
╨B═
!__inference__wrapped_model_689662left_inputs"Ш
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
Ў2є
*__inference_basemodel_layer_call_fn_690958
*__inference_basemodel_layer_call_fn_693471
*__inference_basemodel_layer_call_fn_693526
*__inference_basemodel_layer_call_fn_691698└
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
E__inference_basemodel_layer_call_and_return_conditional_losses_693725
E__inference_basemodel_layer_call_and_return_conditional_losses_694021
E__inference_basemodel_layer_call_and_return_conditional_losses_691838
E__inference_basemodel_layer_call_and_return_conditional_losses_691978└
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
╧B╠
$__inference_signature_wrapper_692759left_inputs"Ф
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
ж2г
4__inference_stream_0_input_drop_layer_call_fn_694026
4__inference_stream_0_input_drop_layer_call_fn_694031┤
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
▄2┘
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_694036
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_694048┤
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
ж2г
4__inference_stream_1_input_drop_layer_call_fn_694053
4__inference_stream_1_input_drop_layer_call_fn_694058┤
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
▄2┘
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_694063
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_694075┤
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
ж2г
4__inference_stream_2_input_drop_layer_call_fn_694080
4__inference_stream_2_input_drop_layer_call_fn_694085┤
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
▄2┘
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_694090
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_694102┤
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
┌2╫
0__inference_stream_0_conv_1_layer_call_fn_694126в
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
ї2Є
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_694156в
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
┌2╫
0__inference_stream_1_conv_1_layer_call_fn_694180в
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
ї2Є
K__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_694210в
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
┌2╫
0__inference_stream_2_conv_1_layer_call_fn_694234в
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
ї2Є
K__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_694264в
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
Т2П
4__inference_batch_normalization_layer_call_fn_694277
4__inference_batch_normalization_layer_call_fn_694290
4__inference_batch_normalization_layer_call_fn_694303
4__inference_batch_normalization_layer_call_fn_694316┤
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
■2√
O__inference_batch_normalization_layer_call_and_return_conditional_losses_694336
O__inference_batch_normalization_layer_call_and_return_conditional_losses_694370
O__inference_batch_normalization_layer_call_and_return_conditional_losses_694390
O__inference_batch_normalization_layer_call_and_return_conditional_losses_694424┤
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
Ъ2Ч
6__inference_batch_normalization_1_layer_call_fn_694437
6__inference_batch_normalization_1_layer_call_fn_694450
6__inference_batch_normalization_1_layer_call_fn_694463
6__inference_batch_normalization_1_layer_call_fn_694476┤
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_694496
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_694530
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_694550
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_694584┤
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
Ъ2Ч
6__inference_batch_normalization_2_layer_call_fn_694597
6__inference_batch_normalization_2_layer_call_fn_694610
6__inference_batch_normalization_2_layer_call_fn_694623
6__inference_batch_normalization_2_layer_call_fn_694636┤
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_694656
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_694690
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_694710
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_694744┤
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
╒2╥
+__inference_activation_layer_call_fn_694749в
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
Ё2э
F__inference_activation_layer_call_and_return_conditional_losses_694754в
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
╫2╘
-__inference_activation_1_layer_call_fn_694759в
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
H__inference_activation_1_layer_call_and_return_conditional_losses_694764в
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
╫2╘
-__inference_activation_2_layer_call_fn_694769в
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
H__inference_activation_2_layer_call_and_return_conditional_losses_694774в
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
Т2П
3__inference_stream_0_maxpool_1_layer_call_fn_694779
3__inference_stream_0_maxpool_1_layer_call_fn_694784в
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
╚2┼
N__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_694792
N__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_694800в
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
Т2П
3__inference_stream_1_maxpool_1_layer_call_fn_694805
3__inference_stream_1_maxpool_1_layer_call_fn_694810в
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
╚2┼
N__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_694818
N__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_694826в
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
Т2П
3__inference_stream_2_maxpool_1_layer_call_fn_694831
3__inference_stream_2_maxpool_1_layer_call_fn_694836в
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
╚2┼
N__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_694844
N__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_694852в
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
Ю2Ы
0__inference_stream_0_drop_1_layer_call_fn_694857
0__inference_stream_0_drop_1_layer_call_fn_694862┤
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
╘2╤
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_694867
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_694879┤
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
Ю2Ы
0__inference_stream_1_drop_1_layer_call_fn_694884
0__inference_stream_1_drop_1_layer_call_fn_694889┤
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
╘2╤
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_694894
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_694906┤
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
Ю2Ы
0__inference_stream_2_drop_1_layer_call_fn_694911
0__inference_stream_2_drop_1_layer_call_fn_694916┤
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
╘2╤
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_694921
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_694933┤
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
л2и
9__inference_global_average_pooling1d_layer_call_fn_694938
9__inference_global_average_pooling1d_layer_call_fn_694943п
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
с2▐
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_694949
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_694955п
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
п2м
;__inference_global_average_pooling1d_1_layer_call_fn_694960
;__inference_global_average_pooling1d_1_layer_call_fn_694965п
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
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_694971
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_694977п
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
п2м
;__inference_global_average_pooling1d_2_layer_call_fn_694982
;__inference_global_average_pooling1d_2_layer_call_fn_694987п
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
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_694993
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_694999п
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
╓2╙
,__inference_concatenate_layer_call_fn_695006в
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
ё2ю
G__inference_concatenate_layer_call_and_return_conditional_losses_695014в
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
Ю2Ы
0__inference_dense_1_dropout_layer_call_fn_695019
0__inference_dense_1_dropout_layer_call_fn_695024┤
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
╘2╤
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_695029
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_695033┤
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
╥2╧
(__inference_dense_1_layer_call_fn_695057в
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
э2ъ
C__inference_dense_1_layer_call_and_return_conditional_losses_695082в
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
к2з
6__inference_batch_normalization_3_layer_call_fn_695095
6__inference_batch_normalization_3_layer_call_fn_695108┤
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_695128
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_695162┤
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
▌2┌
3__inference_dense_activation_1_layer_call_fn_695167в
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
°2ї
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_695171в
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
│2░
__inference_loss_fn_0_695191П
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
│2░
__inference_loss_fn_1_695211П
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
│2░
__inference_loss_fn_2_695231П
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
│2░
__inference_loss_fn_3_695251П
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
annotationsк *в ▓
!__inference__wrapped_model_689662М-.+,)*>3=4<1;2:/9056@7?89в6
/в,
*К'
left_inputs         Ї
к "5к2
0
	basemodel#К 
	basemodel         @о
H__inference_activation_1_layer_call_and_return_conditional_losses_694764b4в1
*в'
%К"
inputs         Ї@
к "*в'
 К
0         Ї@
Ъ Ж
-__inference_activation_1_layer_call_fn_694759U4в1
*в'
%К"
inputs         Ї@
к "К         Ї@о
H__inference_activation_2_layer_call_and_return_conditional_losses_694774b4в1
*в'
%К"
inputs         Ї@
к "*в'
 К
0         Ї@
Ъ Ж
-__inference_activation_2_layer_call_fn_694769U4в1
*в'
%К"
inputs         Ї@
к "К         Ї@м
F__inference_activation_layer_call_and_return_conditional_losses_694754b4в1
*в'
%К"
inputs         Ї@
к "*в'
 К
0         Ї@
Ъ Д
+__inference_activation_layer_call_fn_694749U4в1
*в'
%К"
inputs         Ї@
к "К         Ї@ж
E__inference_basemodel_layer_call_and_return_conditional_losses_691838▄-.+,)*>3=4<1;2:/9056@7?8ШвФ
МвИ
~Ъ{
'К$
inputs_0         Ї
'К$
inputs_1         Ї
'К$
inputs_2         Ї
p 

 
к "%в"
К
0         @
Ъ ж
E__inference_basemodel_layer_call_and_return_conditional_losses_691978▄-.+,)*=>34;<129:/056?@78ШвФ
МвИ
~Ъ{
'К$
inputs_0         Ї
'К$
inputs_1         Ї
'К$
inputs_2         Ї
p

 
к "%в"
К
0         @
Ъ ж
E__inference_basemodel_layer_call_and_return_conditional_losses_693725▄-.+,)*>3=4<1;2:/9056@7?8ШвФ
МвИ
~Ъ{
'К$
inputs/0         Ї
'К$
inputs/1         Ї
'К$
inputs/2         Ї
p 

 
к "%в"
К
0         @
Ъ ж
E__inference_basemodel_layer_call_and_return_conditional_losses_694021▄-.+,)*=>34;<129:/056?@78ШвФ
МвИ
~Ъ{
'К$
inputs/0         Ї
'К$
inputs/1         Ї
'К$
inputs/2         Ї
p

 
к "%в"
К
0         @
Ъ ■
*__inference_basemodel_layer_call_fn_690958╧-.+,)*>3=4<1;2:/9056@7?8ШвФ
МвИ
~Ъ{
'К$
inputs_0         Ї
'К$
inputs_1         Ї
'К$
inputs_2         Ї
p 

 
к "К         @■
*__inference_basemodel_layer_call_fn_691698╧-.+,)*=>34;<129:/056?@78ШвФ
МвИ
~Ъ{
'К$
inputs_0         Ї
'К$
inputs_1         Ї
'К$
inputs_2         Ї
p

 
к "К         @■
*__inference_basemodel_layer_call_fn_693471╧-.+,)*>3=4<1;2:/9056@7?8ШвФ
МвИ
~Ъ{
'К$
inputs/0         Ї
'К$
inputs/1         Ї
'К$
inputs/2         Ї
p 

 
к "К         @■
*__inference_basemodel_layer_call_fn_693526╧-.+,)*=>34;<129:/056?@78ШвФ
МвИ
~Ъ{
'К$
inputs/0         Ї
'К$
inputs/1         Ї
'К$
inputs/2         Ї
p

 
к "К         @╤
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_694496|<1;2@в=
6в3
-К*
inputs                  @
p 
к "2в/
(К%
0                  @
Ъ ╤
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_694530|;<12@в=
6в3
-К*
inputs                  @
p
к "2в/
(К%
0                  @
Ъ ┴
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_694550l<1;28в5
.в+
%К"
inputs         Ї@
p 
к "*в'
 К
0         Ї@
Ъ ┴
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_694584l;<128в5
.в+
%К"
inputs         Ї@
p
к "*в'
 К
0         Ї@
Ъ й
6__inference_batch_normalization_1_layer_call_fn_694437o<1;2@в=
6в3
-К*
inputs                  @
p 
к "%К"                  @й
6__inference_batch_normalization_1_layer_call_fn_694450o;<12@в=
6в3
-К*
inputs                  @
p
к "%К"                  @Щ
6__inference_batch_normalization_1_layer_call_fn_694463_<1;28в5
.в+
%К"
inputs         Ї@
p 
к "К         Ї@Щ
6__inference_batch_normalization_1_layer_call_fn_694476_;<128в5
.в+
%К"
inputs         Ї@
p
к "К         Ї@╤
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_694656|>3=4@в=
6в3
-К*
inputs                  @
p 
к "2в/
(К%
0                  @
Ъ ╤
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_694690|=>34@в=
6в3
-К*
inputs                  @
p
к "2в/
(К%
0                  @
Ъ ┴
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_694710l>3=48в5
.в+
%К"
inputs         Ї@
p 
к "*в'
 К
0         Ї@
Ъ ┴
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_694744l=>348в5
.в+
%К"
inputs         Ї@
p
к "*в'
 К
0         Ї@
Ъ й
6__inference_batch_normalization_2_layer_call_fn_694597o>3=4@в=
6в3
-К*
inputs                  @
p 
к "%К"                  @й
6__inference_batch_normalization_2_layer_call_fn_694610o=>34@в=
6в3
-К*
inputs                  @
p
к "%К"                  @Щ
6__inference_batch_normalization_2_layer_call_fn_694623_>3=48в5
.в+
%К"
inputs         Ї@
p 
к "К         Ї@Щ
6__inference_batch_normalization_2_layer_call_fn_694636_=>348в5
.в+
%К"
inputs         Ї@
p
к "К         Ї@╖
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_695128b@7?83в0
)в&
 К
inputs         @
p 
к "%в"
К
0         @
Ъ ╖
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_695162b?@783в0
)в&
 К
inputs         @
p
к "%в"
К
0         @
Ъ П
6__inference_batch_normalization_3_layer_call_fn_695095U@7?83в0
)в&
 К
inputs         @
p 
к "К         @П
6__inference_batch_normalization_3_layer_call_fn_695108U?@783в0
)в&
 К
inputs         @
p
к "К         @╧
O__inference_batch_normalization_layer_call_and_return_conditional_losses_694336|:/90@в=
6в3
-К*
inputs                  @
p 
к "2в/
(К%
0                  @
Ъ ╧
O__inference_batch_normalization_layer_call_and_return_conditional_losses_694370|9:/0@в=
6в3
-К*
inputs                  @
p
к "2в/
(К%
0                  @
Ъ ┐
O__inference_batch_normalization_layer_call_and_return_conditional_losses_694390l:/908в5
.в+
%К"
inputs         Ї@
p 
к "*в'
 К
0         Ї@
Ъ ┐
O__inference_batch_normalization_layer_call_and_return_conditional_losses_694424l9:/08в5
.в+
%К"
inputs         Ї@
p
к "*в'
 К
0         Ї@
Ъ з
4__inference_batch_normalization_layer_call_fn_694277o:/90@в=
6в3
-К*
inputs                  @
p 
к "%К"                  @з
4__inference_batch_normalization_layer_call_fn_694290o9:/0@в=
6в3
-К*
inputs                  @
p
к "%К"                  @Ч
4__inference_batch_normalization_layer_call_fn_694303_:/908в5
.в+
%К"
inputs         Ї@
p 
к "К         Ї@Ч
4__inference_batch_normalization_layer_call_fn_694316_9:/08в5
.в+
%К"
inputs         Ї@
p
к "К         Ї@Ї
G__inference_concatenate_layer_call_and_return_conditional_losses_695014и~в{
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
Ъ ╠
,__inference_concatenate_layer_call_fn_695006Ы~в{
tвq
oЪl
"К
inputs/0         @
"К
inputs/1         @
"К
inputs/2         @
к "К         └н
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_695029^4в1
*в'
!К
inputs         └
p 
к "&в#
К
0         └
Ъ н
K__inference_dense_1_dropout_layer_call_and_return_conditional_losses_695033^4в1
*в'
!К
inputs         └
p
к "&в#
К
0         └
Ъ Е
0__inference_dense_1_dropout_layer_call_fn_695019Q4в1
*в'
!К
inputs         └
p 
к "К         └Е
0__inference_dense_1_dropout_layer_call_fn_695024Q4в1
*в'
!К
inputs         └
p
к "К         └д
C__inference_dense_1_layer_call_and_return_conditional_losses_695082]560в-
&в#
!К
inputs         └
к "%в"
К
0         @
Ъ |
(__inference_dense_1_layer_call_fn_695057P560в-
&в#
!К
inputs         └
к "К         @к
N__inference_dense_activation_1_layer_call_and_return_conditional_losses_695171X/в,
%в"
 К
inputs         @
к "%в"
К
0         @
Ъ В
3__inference_dense_activation_1_layer_call_fn_695167K/в,
%в"
 К
inputs         @
к "К         @╒
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_694971{IвF
?в<
6К3
inputs'                           

 
к ".в+
$К!
0                  
Ъ ╗
V__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_694977a8в5
.в+
%К"
inputs         ·@

 
к "%в"
К
0         @
Ъ н
;__inference_global_average_pooling1d_1_layer_call_fn_694960nIвF
?в<
6К3
inputs'                           

 
к "!К                  У
;__inference_global_average_pooling1d_1_layer_call_fn_694965T8в5
.в+
%К"
inputs         ·@

 
к "К         @╒
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_694993{IвF
?в<
6К3
inputs'                           

 
к ".в+
$К!
0                  
Ъ ╗
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_694999a8в5
.в+
%К"
inputs         ·@

 
к "%в"
К
0         @
Ъ н
;__inference_global_average_pooling1d_2_layer_call_fn_694982nIвF
?в<
6К3
inputs'                           

 
к "!К                  У
;__inference_global_average_pooling1d_2_layer_call_fn_694987T8в5
.в+
%К"
inputs         ·@

 
к "К         @╙
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_694949{IвF
?в<
6К3
inputs'                           

 
к ".в+
$К!
0                  
Ъ ╣
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_694955a8в5
.в+
%К"
inputs         ·@

 
к "%в"
К
0         @
Ъ л
9__inference_global_average_pooling1d_layer_call_fn_694938nIвF
?в<
6К3
inputs'                           

 
к "!К                  С
9__inference_global_average_pooling1d_layer_call_fn_694943T8в5
.в+
%К"
inputs         ·@

 
к "К         @;
__inference_loss_fn_0_695191)в

в 
к "К ;
__inference_loss_fn_1_695211+в

в 
к "К ;
__inference_loss_fn_2_695231-в

в 
к "К ;
__inference_loss_fn_3_6952515в

в 
к "К ╠
C__inference_model_1_layer_call_and_return_conditional_losses_692531Д-.+,)*>3=4<1;2:/9056@7?8Aв>
7в4
*К'
left_inputs         Ї
p 

 
к "%в"
К
0         @
Ъ ╠
C__inference_model_1_layer_call_and_return_conditional_losses_692644Д-.+,)*=>34;<129:/056?@78Aв>
7в4
*К'
left_inputs         Ї
p

 
к "%в"
К
0         @
Ъ ╞
C__inference_model_1_layer_call_and_return_conditional_losses_693062-.+,)*>3=4<1;2:/9056@7?8<в9
2в/
%К"
inputs         Ї
p 

 
к "%в"
К
0         @
Ъ ╞
C__inference_model_1_layer_call_and_return_conditional_losses_693356-.+,)*=>34;<129:/056?@78<в9
2в/
%К"
inputs         Ї
p

 
к "%в"
К
0         @
Ъ г
(__inference_model_1_layer_call_fn_692146w-.+,)*>3=4<1;2:/9056@7?8Aв>
7в4
*К'
left_inputs         Ї
p 

 
к "К         @г
(__inference_model_1_layer_call_fn_692418w-.+,)*=>34;<129:/056?@78Aв>
7в4
*К'
left_inputs         Ї
p

 
к "К         @Ю
(__inference_model_1_layer_call_fn_692812r-.+,)*>3=4<1;2:/9056@7?8<в9
2в/
%К"
inputs         Ї
p 

 
к "К         @Ю
(__inference_model_1_layer_call_fn_692865r-.+,)*=>34;<129:/056?@78<в9
2в/
%К"
inputs         Ї
p

 
к "К         @─
$__inference_signature_wrapper_692759Ы-.+,)*>3=4<1;2:/9056@7?8HвE
в 
>к;
9
left_inputs*К'
left_inputs         Ї"5к2
0
	basemodel#К 
	basemodel         @╡
K__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_694156f)*4в1
*в'
%К"
inputs         Ї
к "*в'
 К
0         Ї@
Ъ Н
0__inference_stream_0_conv_1_layer_call_fn_694126Y)*4в1
*в'
%К"
inputs         Ї
к "К         Ї@╡
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_694867f8в5
.в+
%К"
inputs         ·@
p 
к "*в'
 К
0         ·@
Ъ ╡
K__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_694879f8в5
.в+
%К"
inputs         ·@
p
к "*в'
 К
0         ·@
Ъ Н
0__inference_stream_0_drop_1_layer_call_fn_694857Y8в5
.в+
%К"
inputs         ·@
p 
к "К         ·@Н
0__inference_stream_0_drop_1_layer_call_fn_694862Y8в5
.в+
%К"
inputs         ·@
p
к "К         ·@╣
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_694036f8в5
.в+
%К"
inputs         Ї
p 
к "*в'
 К
0         Ї
Ъ ╣
O__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_694048f8в5
.в+
%К"
inputs         Ї
p
к "*в'
 К
0         Ї
Ъ С
4__inference_stream_0_input_drop_layer_call_fn_694026Y8в5
.в+
%К"
inputs         Ї
p 
к "К         ЇС
4__inference_stream_0_input_drop_layer_call_fn_694031Y8в5
.в+
%К"
inputs         Ї
p
к "К         Ї╫
N__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_694792ДEвB
;в8
6К3
inputs'                           
к ";в8
1К.
0'                           
Ъ ┤
N__inference_stream_0_maxpool_1_layer_call_and_return_conditional_losses_694800b4в1
*в'
%К"
inputs         Ї@
к "*в'
 К
0         ·@
Ъ о
3__inference_stream_0_maxpool_1_layer_call_fn_694779wEвB
;в8
6К3
inputs'                           
к ".К+'                           М
3__inference_stream_0_maxpool_1_layer_call_fn_694784U4в1
*в'
%К"
inputs         Ї@
к "К         ·@╡
K__inference_stream_1_conv_1_layer_call_and_return_conditional_losses_694210f+,4в1
*в'
%К"
inputs         Ї
к "*в'
 К
0         Ї@
Ъ Н
0__inference_stream_1_conv_1_layer_call_fn_694180Y+,4в1
*в'
%К"
inputs         Ї
к "К         Ї@╡
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_694894f8в5
.в+
%К"
inputs         ·@
p 
к "*в'
 К
0         ·@
Ъ ╡
K__inference_stream_1_drop_1_layer_call_and_return_conditional_losses_694906f8в5
.в+
%К"
inputs         ·@
p
к "*в'
 К
0         ·@
Ъ Н
0__inference_stream_1_drop_1_layer_call_fn_694884Y8в5
.в+
%К"
inputs         ·@
p 
к "К         ·@Н
0__inference_stream_1_drop_1_layer_call_fn_694889Y8в5
.в+
%К"
inputs         ·@
p
к "К         ·@╣
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_694063f8в5
.в+
%К"
inputs         Ї
p 
к "*в'
 К
0         Ї
Ъ ╣
O__inference_stream_1_input_drop_layer_call_and_return_conditional_losses_694075f8в5
.в+
%К"
inputs         Ї
p
к "*в'
 К
0         Ї
Ъ С
4__inference_stream_1_input_drop_layer_call_fn_694053Y8в5
.в+
%К"
inputs         Ї
p 
к "К         ЇС
4__inference_stream_1_input_drop_layer_call_fn_694058Y8в5
.в+
%К"
inputs         Ї
p
к "К         Ї╫
N__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_694818ДEвB
;в8
6К3
inputs'                           
к ";в8
1К.
0'                           
Ъ ┤
N__inference_stream_1_maxpool_1_layer_call_and_return_conditional_losses_694826b4в1
*в'
%К"
inputs         Ї@
к "*в'
 К
0         ·@
Ъ о
3__inference_stream_1_maxpool_1_layer_call_fn_694805wEвB
;в8
6К3
inputs'                           
к ".К+'                           М
3__inference_stream_1_maxpool_1_layer_call_fn_694810U4в1
*в'
%К"
inputs         Ї@
к "К         ·@╡
K__inference_stream_2_conv_1_layer_call_and_return_conditional_losses_694264f-.4в1
*в'
%К"
inputs         Ї
к "*в'
 К
0         Ї@
Ъ Н
0__inference_stream_2_conv_1_layer_call_fn_694234Y-.4в1
*в'
%К"
inputs         Ї
к "К         Ї@╡
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_694921f8в5
.в+
%К"
inputs         ·@
p 
к "*в'
 К
0         ·@
Ъ ╡
K__inference_stream_2_drop_1_layer_call_and_return_conditional_losses_694933f8в5
.в+
%К"
inputs         ·@
p
к "*в'
 К
0         ·@
Ъ Н
0__inference_stream_2_drop_1_layer_call_fn_694911Y8в5
.в+
%К"
inputs         ·@
p 
к "К         ·@Н
0__inference_stream_2_drop_1_layer_call_fn_694916Y8в5
.в+
%К"
inputs         ·@
p
к "К         ·@╣
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_694090f8в5
.в+
%К"
inputs         Ї
p 
к "*в'
 К
0         Ї
Ъ ╣
O__inference_stream_2_input_drop_layer_call_and_return_conditional_losses_694102f8в5
.в+
%К"
inputs         Ї
p
к "*в'
 К
0         Ї
Ъ С
4__inference_stream_2_input_drop_layer_call_fn_694080Y8в5
.в+
%К"
inputs         Ї
p 
к "К         ЇС
4__inference_stream_2_input_drop_layer_call_fn_694085Y8в5
.в+
%К"
inputs         Ї
p
к "К         Ї╫
N__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_694844ДEвB
;в8
6К3
inputs'                           
к ";в8
1К.
0'                           
Ъ ┤
N__inference_stream_2_maxpool_1_layer_call_and_return_conditional_losses_694852b4в1
*в'
%К"
inputs         Ї@
к "*в'
 К
0         ·@
Ъ о
3__inference_stream_2_maxpool_1_layer_call_fn_694831wEвB
;в8
6К3
inputs'                           
к ".К+'                           М
3__inference_stream_2_maxpool_1_layer_call_fn_694836U4в1
*в'
%К"
inputs         Ї@
к "К         ·@