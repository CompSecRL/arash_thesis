╙Ы.
▐┤
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
 И"serve*2.6.22v2.6.1-9-gc2363d6d0258┴л+
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
shape: *'
shared_namestream_0_conv_1/kernel
Е
*stream_0_conv_1/kernel/Read/ReadVariableOpReadVariableOpstream_0_conv_1/kernel*"
_output_shapes
: *
dtype0
А
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
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namebatch_normalization/gamma
Г
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0
М
stream_0_conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_namestream_0_conv_2/kernel
Е
*stream_0_conv_2/kernel/Read/ReadVariableOpReadVariableOpstream_0_conv_2/kernel*"
_output_shapes
: @*
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
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:T*
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:T*
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance
Ч
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
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
shape:T*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:T*
dtype0
в
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*6
shared_name'%batch_normalization_2/moving_variance
Ы
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
Ъ
Adam/stream_0_conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdam/stream_0_conv_1/kernel/m
У
1Adam/stream_0_conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/kernel/m*"
_output_shapes
: *
dtype0
О
Adam/stream_0_conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdam/stream_0_conv_1/bias/m
З
/Adam/stream_0_conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/bias/m*
_output_shapes
: *
dtype0
Ш
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/batch_normalization/gamma/m
С
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
: *
dtype0
Ц
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/batch_normalization/beta/m
П
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
: *
dtype0
Ъ
Adam/stream_0_conv_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*.
shared_nameAdam/stream_0_conv_2/kernel/m
У
1Adam/stream_0_conv_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_2/kernel/m*"
_output_shapes
: @*
dtype0
О
Adam/stream_0_conv_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/stream_0_conv_2/bias/m
З
/Adam/stream_0_conv_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_2/bias/m*
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
Ж
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
Ь
"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*3
shared_name$"Adam/batch_normalization_2/gamma/m
Х
6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes
:T*
dtype0
Ъ
!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*2
shared_name#!Adam/batch_normalization_2/beta/m
У
5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes
:T*
dtype0
Ъ
Adam/stream_0_conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdam/stream_0_conv_1/kernel/v
У
1Adam/stream_0_conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/kernel/v*"
_output_shapes
: *
dtype0
О
Adam/stream_0_conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdam/stream_0_conv_1/bias/v
З
/Adam/stream_0_conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_1/bias/v*
_output_shapes
: *
dtype0
Ш
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/batch_normalization/gamma/v
С
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
: *
dtype0
Ц
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/batch_normalization/beta/v
П
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
: *
dtype0
Ъ
Adam/stream_0_conv_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*.
shared_nameAdam/stream_0_conv_2/kernel/v
У
1Adam/stream_0_conv_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_2/kernel/v*"
_output_shapes
: @*
dtype0
О
Adam/stream_0_conv_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/stream_0_conv_2/bias/v
З
/Adam/stream_0_conv_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/stream_0_conv_2/bias/v*
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
Ж
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
Ь
"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*3
shared_name$"Adam/batch_normalization_2/gamma/v
Х
6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes
:T*
dtype0
Ъ
!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*2
shared_name#!Adam/batch_normalization_2/beta/v
У
5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes
:T*
dtype0

NoOpNoOp
Щ`
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╘_
value╩_B╟_ B└_
┐
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
╢
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
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer-14
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
 	variables
!	keras_api
░

"beta_1

#beta_2
	$decay
%learning_rate
&iter'm╬(m╧)m╨*m╤+m╥,m╙-m╘.m╒/m╓0m╫1m╪2m┘'v┌(v█)v▄*v▌+v▐,v▀-vр.vс/vт0vу1vф2vх
 
V
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
Ж
'0
(1
)2
*3
34
45
+6
,7
-8
.9
510
611
/12
013
114
215
716
817
н
9metrics

:layers
;layer_metrics
<layer_regularization_losses
regularization_losses
trainable_variables
=non_trainable_variables
	variables
 
 
R
>regularization_losses
?trainable_variables
@	variables
A	keras_api
h

'kernel
(bias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
Ч
Faxis
	)gamma
*beta
3moving_mean
4moving_variance
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
R
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
R
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
h

+kernel
,bias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
Ч
Waxis
	-gamma
.beta
5moving_mean
6moving_variance
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
R
\regularization_losses
]trainable_variables
^	variables
_	keras_api
R
`regularization_losses
atrainable_variables
b	variables
c	keras_api
R
dregularization_losses
etrainable_variables
f	variables
g	keras_api
R
hregularization_losses
itrainable_variables
j	variables
k	keras_api
h

/kernel
0bias
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
Ч
paxis
	1gamma
2beta
7moving_mean
8moving_variance
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
R
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
 
V
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
Ж
'0
(1
)2
*3
34
45
+6
,7
-8
.9
510
611
/12
013
114
215
716
817
н
ymetrics

zlayers
{layer_metrics
|layer_regularization_losses
regularization_losses
trainable_variables
}non_trainable_variables
	variables
 
 
 
░
~metrics

layers
Аlayer_metrics
 Бlayer_regularization_losses
regularization_losses
trainable_variables
Вnon_trainable_variables
 	variables
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

Г0

0
1
2
3
 
 
*
30
41
52
63
74
85
 
 
 
▓
Дmetrics
Еlayers
Жlayer_metrics
 Зlayer_regularization_losses
>regularization_losses
?trainable_variables
Иnon_trainable_variables
@	variables
 

'0
(1

'0
(1
▓
Йmetrics
Кlayers
Лlayer_metrics
 Мlayer_regularization_losses
Bregularization_losses
Ctrainable_variables
Нnon_trainable_variables
D	variables
 
 

)0
*1

)0
*1
32
43
▓
Оmetrics
Пlayers
Рlayer_metrics
 Сlayer_regularization_losses
Gregularization_losses
Htrainable_variables
Тnon_trainable_variables
I	variables
 
 
 
▓
Уmetrics
Фlayers
Хlayer_metrics
 Цlayer_regularization_losses
Kregularization_losses
Ltrainable_variables
Чnon_trainable_variables
M	variables
 
 
 
▓
Шmetrics
Щlayers
Ъlayer_metrics
 Ыlayer_regularization_losses
Oregularization_losses
Ptrainable_variables
Ьnon_trainable_variables
Q	variables
 

+0
,1

+0
,1
▓
Эmetrics
Юlayers
Яlayer_metrics
 аlayer_regularization_losses
Sregularization_losses
Ttrainable_variables
бnon_trainable_variables
U	variables
 
 

-0
.1

-0
.1
52
63
▓
вmetrics
гlayers
дlayer_metrics
 еlayer_regularization_losses
Xregularization_losses
Ytrainable_variables
жnon_trainable_variables
Z	variables
 
 
 
▓
зmetrics
иlayers
йlayer_metrics
 кlayer_regularization_losses
\regularization_losses
]trainable_variables
лnon_trainable_variables
^	variables
 
 
 
▓
мmetrics
нlayers
оlayer_metrics
 пlayer_regularization_losses
`regularization_losses
atrainable_variables
░non_trainable_variables
b	variables
 
 
 
▓
▒metrics
▓layers
│layer_metrics
 ┤layer_regularization_losses
dregularization_losses
etrainable_variables
╡non_trainable_variables
f	variables
 
 
 
▓
╢metrics
╖layers
╕layer_metrics
 ╣layer_regularization_losses
hregularization_losses
itrainable_variables
║non_trainable_variables
j	variables
 

/0
01

/0
01
▓
╗metrics
╝layers
╜layer_metrics
 ╛layer_regularization_losses
lregularization_losses
mtrainable_variables
┐non_trainable_variables
n	variables
 
 

10
21

10
21
72
83
▓
└metrics
┴layers
┬layer_metrics
 ├layer_regularization_losses
qregularization_losses
rtrainable_variables
─non_trainable_variables
s	variables
 
 
 
▓
┼metrics
╞layers
╟layer_metrics
 ╚layer_regularization_losses
uregularization_losses
vtrainable_variables
╔non_trainable_variables
w	variables
 
n
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
 
 
*
30
41
52
63
74
85
 
 
 
 
 
8

╩total

╦count
╠	variables
═	keras_api
 
 
 
 
 
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
30
41
 
 
 
 
 
 
 
 
 
 
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
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
70
81
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
╩0
╦1

╠	variables
}
VARIABLE_VALUEAdam/stream_0_conv_1/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_0_conv_1/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/batch_normalization/gamma/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/batch_normalization/beta/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/stream_0_conv_2/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_0_conv_2/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_1/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_1/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE!Adam/batch_normalization_2/beta/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/stream_0_conv_1/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_0_conv_1/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/batch_normalization/gamma/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/batch_normalization/beta/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/stream_0_conv_2/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/stream_0_conv_2/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_1/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_1/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE!Adam/batch_normalization_2/beta/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
И
serving_default_left_inputsPlaceholder*,
_output_shapes
:         ╓*
dtype0*!
shape:         ╓
Й
serving_default_right_inputsPlaceholder*,
_output_shapes
:         ╓*
dtype0*!
shape:         ╓
°
StatefulPartitionedCallStatefulPartitionedCallserving_default_left_inputsserving_default_right_inputsstream_0_conv_1/kernelstream_0_conv_1/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betastream_0_conv_2/kernelstream_0_conv_2/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betadense_1/kerneldense_1/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/beta*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *.
f)R'
%__inference_signature_wrapper_5261425
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╚
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
GPU2*0J 8В *)
f$R"
 __inference__traced_save_5263666
я
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
GPU2*0J 8В *,
f'R%
#__inference__traced_restore_5263823┐╢)
Щ;
╦
B__inference_model_layer_call_and_return_conditional_losses_5261277
left_inputs
right_inputs'
basemodel_5261201: 
basemodel_5261203: 
basemodel_5261205: 
basemodel_5261207: 
basemodel_5261209: 
basemodel_5261211: '
basemodel_5261213: @
basemodel_5261215:@
basemodel_5261217:@
basemodel_5261219:@
basemodel_5261221:@
basemodel_5261223:@#
basemodel_5261225:@T
basemodel_5261227:T
basemodel_5261229:T
basemodel_5261231:T
basemodel_5261233:T
basemodel_5261235:T
identityИв!basemodel/StatefulPartitionedCallв#basemodel/StatefulPartitionedCall_1в-dense_1/kernel/Regularizer/Abs/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpЇ
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsbasemodel_5261201basemodel_5261203basemodel_5261205basemodel_5261207basemodel_5261209basemodel_5261211basemodel_5261213basemodel_5261215basemodel_5261217basemodel_5261219basemodel_5261221basemodel_5261223basemodel_5261225basemodel_5261227basemodel_5261229basemodel_5261231basemodel_5261233basemodel_5261235*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_52605782#
!basemodel/StatefulPartitionedCall∙
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallright_inputsbasemodel_5261201basemodel_5261203basemodel_5261205basemodel_5261207basemodel_5261209basemodel_5261211basemodel_5261213basemodel_5261215basemodel_5261217basemodel_5261219basemodel_5261221basemodel_5261223basemodel_5261225basemodel_5261227basemodel_5261229basemodel_5261231basemodel_5261233basemodel_5261235*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_52605782%
#basemodel/StatefulPartitionedCall_1л
distance/PartitionedCallPartitionedCall*basemodel/StatefulPartitionedCall:output:0,basemodel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_distance_layer_call_and_return_conditional_losses_52606472
distance/PartitionedCall─
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_5261201*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
&stream_0_conv_1/kernel/Regularizer/mul╩
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_5261213*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul░
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_5261225*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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
dense_1/kernel/Regularizer/mul|
IdentityIdentity!distance/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity╗
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:         ╓:         ╓: : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:Y U
,
_output_shapes
:         ╓
%
_user_specified_nameleft_inputs:ZV
,
_output_shapes
:         ╓
&
_user_specified_nameright_inputs
∙
j
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_5259761

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         @2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Ї
Ц
)__inference_dense_1_layer_call_fn_5263372

inputs
unknown:@T
	unknown_0:T
identityИвStatefulPartitionedCallў
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
GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_52597792
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
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
у
M
1__inference_stream_0_drop_1_layer_call_fn_5263054

inputs
identity╥
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_52596772
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ╓ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓ :T P
,
_output_shapes
:         ╓ 
 
_user_specified_nameinputs
у
M
1__inference_stream_0_drop_2_layer_call_fn_5263287

inputs
identity╥
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_52597472
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ╓@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓@:T P
,
_output_shapes
:         ╓@
 
_user_specified_nameinputs
Ъ
┐
+__inference_basemodel_layer_call_fn_5262681

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
identityИвStatefulPartitionedCall╩
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
:         T*.
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_52602392
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
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ╓: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ╓
 
_user_specified_nameinputs
▄
╥
7__inference_batch_normalization_2_layer_call_fn_5263439

inputs
unknown:T
	unknown_0:T
	unknown_1:T
	unknown_2:T
identityИвStatefulPartitionedCallЯ
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
GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_52594572
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
з
╦
'__inference_model_layer_call_fn_5261971
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
identityИвStatefulPartitionedCall┘
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
:         *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_52606682
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:         ╓:         ╓: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:         ╓
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:         ╓
"
_user_specified_name
inputs/1
в╗
К
F__inference_basemodel_layer_call_and_return_conditional_losses_5262599
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
identityИв#batch_normalization/AssignMovingAvgв2batch_normalization/AssignMovingAvg/ReadVariableOpв%batch_normalization/AssignMovingAvg_1в4batch_normalization/AssignMovingAvg_1/ReadVariableOpв,batch_normalization/batchnorm/ReadVariableOpв0batch_normalization/batchnorm/mul/ReadVariableOpв%batch_normalization_1/AssignMovingAvgв4batch_normalization_1/AssignMovingAvg/ReadVariableOpв'batch_normalization_1/AssignMovingAvg_1в6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpв.batch_normalization_1/batchnorm/ReadVariableOpв2batch_normalization_1/batchnorm/mul/ReadVariableOpв%batch_normalization_2/AssignMovingAvgв4batch_normalization_2/AssignMovingAvg/ReadVariableOpв'batch_normalization_2/AssignMovingAvg_1в6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpв.batch_normalization_2/batchnorm/ReadVariableOpв2batch_normalization_2/batchnorm/mul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв&stream_0_conv_1/BiasAdd/ReadVariableOpв2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв&stream_0_conv_2/BiasAdd/ReadVariableOpв2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpЛ
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
:         ╓2!
stream_0_input_drop/dropout/Mul~
!stream_0_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/ShapeР
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:         ╓*
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
:         ╓2*
(stream_0_input_drop/dropout/GreaterEqual└
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓2"
 stream_0_input_drop/dropout/Cast╧
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:         ╓2#
!stream_0_input_drop/dropout/Mul_1Щ
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
:         ╓2#
!stream_0_conv_1/conv1d/ExpandDimsш
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2%
#stream_0_conv_1/conv1d/ExpandDims_1ў
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓ *
paddingSAME*
strides
2
stream_0_conv_1/conv1d├
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ╓ *
squeeze_dims

¤        2 
stream_0_conv_1/conv1d/Squeeze╝
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp═
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓ 2
stream_0_conv_1/BiasAdd╣
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indicesщ
 batch_normalization/moments/meanMean stream_0_conv_1/BiasAdd:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2"
 batch_normalization/moments/mean╝
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
: 2*
(batch_normalization/moments/StopGradient 
-batch_normalization/moments/SquaredDifferenceSquaredDifference stream_0_conv_1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:         ╓ 2/
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
: *
	keep_dims(2&
$batch_normalization/moments/variance╜
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2%
#batch_normalization/moments/Squeeze┼
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
: *
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
: *
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpш
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/sub▀
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2)
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
: *
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOpЁ
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/subч
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2+
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
: 2#
!batch_normalization/batchnorm/addЯ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp╒
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/mul╤
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓ 2%
#batch_normalization/batchnorm/mul_1╦
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/mul_2╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02.
,batch_normalization/batchnorm/ReadVariableOp╤
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/sub┌
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ╓ 2%
#batch_normalization/batchnorm/add_1К
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ╓ 2
activation/ReluГ
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
:         ╓ 2
stream_0_drop_1/dropout/MulЛ
stream_0_drop_1/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/ShapeД
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:         ╓ *
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
:         ╓ 2&
$stream_0_drop_1/dropout/GreaterEqual┤
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓ 2
stream_0_drop_1/dropout/Cast┐
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:         ╓ 2
stream_0_drop_1/dropout/Mul_1Щ
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2'
%stream_0_conv_2/conv1d/ExpandDims/dimт
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/dropout/Mul_1:z:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ╓ 2#
!stream_0_conv_2/conv1d/ExpandDimsш
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dimў
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#stream_0_conv_2/conv1d/ExpandDims_1ў
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓@*
paddingSAME*
strides
2
stream_0_conv_2/conv1d├
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:         ╓@*
squeeze_dims

¤        2 
stream_0_conv_2/conv1d/Squeeze╝
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp═
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓@2
stream_0_conv_2/BiasAdd╜
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indicesя
"batch_normalization_1/moments/meanMean stream_0_conv_2/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
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
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_0_conv_2/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:         ╓@21
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
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓@2'
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
:         ╓@2'
%batch_normalization_1/batchnorm/add_1Р
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ╓@2
activation_1/ReluГ
stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
stream_0_drop_2/dropout/Const┴
stream_0_drop_2/dropout/MulMulactivation_1/Relu:activations:0&stream_0_drop_2/dropout/Const:output:0*
T0*,
_output_shapes
:         ╓@2
stream_0_drop_2/dropout/MulН
stream_0_drop_2/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_2/dropout/ShapeД
4stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_2/dropout/Shape:output:0*
T0*,
_output_shapes
:         ╓@*
dtype0*
seed╖*
seed2╖26
4stream_0_drop_2/dropout/random_uniform/RandomUniformХ
&stream_0_drop_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2(
&stream_0_drop_2/dropout/GreaterEqual/yГ
$stream_0_drop_2/dropout/GreaterEqualGreaterEqual=stream_0_drop_2/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ╓@2&
$stream_0_drop_2/dropout/GreaterEqual┤
stream_0_drop_2/dropout/CastCast(stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓@2
stream_0_drop_2/dropout/Cast┐
stream_0_drop_2/dropout/Mul_1Mulstream_0_drop_2/dropout/Mul:z:0 stream_0_drop_2/dropout/Cast:y:0*
T0*,
_output_shapes
:         ╓@2
stream_0_drop_2/dropout/Mul_1д
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices╒
global_average_pooling1d/MeanMean!stream_0_drop_2/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2
global_average_pooling1d/MeanГ
dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dense_1_dropout/dropout/Const├
dense_1_dropout/dropout/MulMul&global_average_pooling1d/Mean:output:0&dense_1_dropout/dropout/Const:output:0*
T0*'
_output_shapes
:         @2
dense_1_dropout/dropout/MulФ
dense_1_dropout/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
:2
dense_1_dropout/dropout/Shapeё
4dense_1_dropout/dropout/random_uniform/RandomUniformRandomUniform&dense_1_dropout/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seed╖26
4dense_1_dropout/dropout/random_uniform/RandomUniformХ
&dense_1_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2(
&dense_1_dropout/dropout/GreaterEqual/y■
$dense_1_dropout/dropout/GreaterEqualGreaterEqual=dense_1_dropout/dropout/random_uniform/RandomUniform:output:0/dense_1_dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2&
$dense_1_dropout/dropout/GreaterEqualп
dense_1_dropout/dropout/CastCast(dense_1_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dense_1_dropout/dropout/Cast║
dense_1_dropout/dropout/Mul_1Muldense_1_dropout/dropout/Mul:z:0 dense_1_dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dense_1_dropout/dropout/Mul_1е
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02
dense_1/MatMul/ReadVariableOpж
dense_1/MatMulMatMul!dense_1_dropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
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
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_2/moments/mean/reduction_indicesу
"batch_normalization_2/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2$
"batch_normalization_2/moments/mean╛
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:T2,
*batch_normalization_2/moments/StopGradient°
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:         T21
/batch_normalization_2/moments/SquaredDifference╛
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_2/moments/variance/reduction_indicesК
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2(
&batch_normalization_2/moments/variance┬
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze╩
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2)
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
:T*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpЁ
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_2/AssignMovingAvg/subч
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2+
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
:T*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp°
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_2/AssignMovingAvg_1/subя
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2-
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
:T2%
#batch_normalization_2/batchnorm/addе
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp▌
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/mul╩
%batch_normalization_2/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_2/batchnorm/mul_1╙
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/mul_2╘
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp┘
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/sub▌
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_2/batchnorm/add_1а
dense_activation_1/SigmoidSigmoid)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2
dense_activation_1/Sigmoidю
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul┼
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

Identity┌

NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ╓: : : : : : : : : : : : : : : : : : 2J
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
:         ╓
"
_user_specified_name
inputs/0
№:
┬
B__inference_model_layer_call_and_return_conditional_losses_5260668

inputs
inputs_1'
basemodel_5260579: 
basemodel_5260581: 
basemodel_5260583: 
basemodel_5260585: 
basemodel_5260587: 
basemodel_5260589: '
basemodel_5260591: @
basemodel_5260593:@
basemodel_5260595:@
basemodel_5260597:@
basemodel_5260599:@
basemodel_5260601:@#
basemodel_5260603:@T
basemodel_5260605:T
basemodel_5260607:T
basemodel_5260609:T
basemodel_5260611:T
basemodel_5260613:T
identityИв!basemodel/StatefulPartitionedCallв#basemodel/StatefulPartitionedCall_1в-dense_1/kernel/Regularizer/Abs/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpя
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsbasemodel_5260579basemodel_5260581basemodel_5260583basemodel_5260585basemodel_5260587basemodel_5260589basemodel_5260591basemodel_5260593basemodel_5260595basemodel_5260597basemodel_5260599basemodel_5260601basemodel_5260603basemodel_5260605basemodel_5260607basemodel_5260609basemodel_5260611basemodel_5260613*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_52605782#
!basemodel/StatefulPartitionedCallї
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1basemodel_5260579basemodel_5260581basemodel_5260583basemodel_5260585basemodel_5260587basemodel_5260589basemodel_5260591basemodel_5260593basemodel_5260595basemodel_5260597basemodel_5260599basemodel_5260601basemodel_5260603basemodel_5260605basemodel_5260607basemodel_5260609basemodel_5260611basemodel_5260613*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_52605782%
#basemodel/StatefulPartitionedCall_1л
distance/PartitionedCallPartitionedCall*basemodel/StatefulPartitionedCall:output:0,basemodel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_distance_layer_call_and_return_conditional_losses_52606472
distance/PartitionedCall─
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_5260579*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
&stream_0_conv_1/kernel/Regularizer/mul╩
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_5260591*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul░
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_5260603*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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
dense_1/kernel/Regularizer/mul|
IdentityIdentity!distance/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity╗
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:         ╓:         ╓: : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:         ╓
 
_user_specified_nameinputs:TP
,
_output_shapes
:         ╓
 
_user_specified_nameinputs
Ж+
щ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5262970

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
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
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         ╓ 2
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
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
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
╫#<2
AssignMovingAvg/decayд
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
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
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
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
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ╓ 2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ╓ 2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ╓ : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ╓ 
 
_user_specified_nameinputs
а
┐
+__inference_basemodel_layer_call_fn_5262640

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
identityИвStatefulPartitionedCall╨
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
:         T*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_52598202
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
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ╓: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ╓
 
_user_specified_nameinputs
т
╙
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_5259630

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
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
:         ╓2
conv1d/ExpandDims╕
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
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓ *
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ╓ *
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓ 2	
BiasAdd▐
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
:         ╓ 2

Identity─
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╓: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:         ╓
 
_user_specified_nameinputs
╝
╥
'__inference_model_layer_call_fn_5260707
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
identityИвStatefulPartitionedCallр
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
:         *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_52606682
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:         ╓:         ╓: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:         ╓
%
_user_specified_nameleft_inputs:ZV
,
_output_shapes
:         ╓
&
_user_specified_nameright_inputs
лd
О

F__inference_basemodel_layer_call_and_return_conditional_losses_5259820

inputs-
stream_0_conv_1_5259631: %
stream_0_conv_1_5259633: )
batch_normalization_5259656: )
batch_normalization_5259658: )
batch_normalization_5259660: )
batch_normalization_5259662: -
stream_0_conv_2_5259701: @%
stream_0_conv_2_5259703:@+
batch_normalization_1_5259726:@+
batch_normalization_1_5259728:@+
batch_normalization_1_5259730:@+
batch_normalization_1_5259732:@!
dense_1_5259780:@T
dense_1_5259782:T+
batch_normalization_2_5259785:T+
batch_normalization_2_5259787:T+
batch_normalization_2_5259789:T+
batch_normalization_2_5259791:T
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв'stream_0_conv_1/StatefulPartitionedCallв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв'stream_0_conv_2/StatefulPartitionedCallв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp■
#stream_0_input_drop/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_52596072%
#stream_0_input_drop/PartitionedCallш
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_5259631stream_0_conv_1_5259633*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_52596302)
'stream_0_conv_1/StatefulPartitionedCall╛
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_5259656batch_normalization_5259658batch_normalization_5259660batch_normalization_5259662*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_52596552-
+batch_normalization/StatefulPartitionedCallС
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_52596702
activation/PartitionedCallП
stream_0_drop_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_52596772!
stream_0_drop_1/PartitionedCallф
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall(stream_0_drop_1/PartitionedCall:output:0stream_0_conv_2_5259701stream_0_conv_2_5259703*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_52597002)
'stream_0_conv_2/StatefulPartitionedCall╠
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_5259726batch_normalization_1_5259728batch_normalization_1_5259730batch_normalization_1_5259732*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_52597252/
-batch_normalization_1/StatefulPartitionedCallЩ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_52597402
activation_1/PartitionedCallС
stream_0_drop_2/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_52597472!
stream_0_drop_2/PartitionedCallк
(global_average_pooling1d/PartitionedCallPartitionedCall(stream_0_drop_2/PartitionedCall:output:0*
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
GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_52597542*
(global_average_pooling1d/PartitionedCallШ
dense_1_dropout/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
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
GPU2*0J 8В *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_52597612!
dense_1_dropout/PartitionedCall╖
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_5259780dense_1_5259782*
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
GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_52597792!
dense_1/StatefulPartitionedCall┐
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_5259785batch_normalization_2_5259787batch_normalization_2_5259789batch_normalization_2_5259791*
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
GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_52594572/
-batch_normalization_2/StatefulPartitionedCallж
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_52597992$
"dense_activation_1/PartitionedCall╩
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_5259631*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
&stream_0_conv_1/kernel/Regularizer/mul╨
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_5259701*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulо
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_5259780*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

Identityї
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ╓: : : : : : : : : : : : : : : : : : 2Z
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
:         ╓
 
_user_specified_nameinputs
Г
╓
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_5263086

inputsA
+conv1d_expanddims_1_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpy
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
:         ╓ 2
conv1d/ExpandDims╕
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
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ╓@*
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
:         ╓@2	
BiasAddф
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         ╓@2

Identity╟
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╓ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:         ╓ 
 
_user_specified_nameinputs
Ў
k
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_5263282

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ╓@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ╓@*
dtype0*
seed╖*
seed2╖2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ╓@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ╓@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ╓@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓@:T P
,
_output_shapes
:         ╓@
 
_user_specified_nameinputs
╫k
║
F__inference_basemodel_layer_call_and_return_conditional_losses_5260239

inputs-
stream_0_conv_1_5260171: %
stream_0_conv_1_5260173: )
batch_normalization_5260176: )
batch_normalization_5260178: )
batch_normalization_5260180: )
batch_normalization_5260182: -
stream_0_conv_2_5260187: @%
stream_0_conv_2_5260189:@+
batch_normalization_1_5260192:@+
batch_normalization_1_5260194:@+
batch_normalization_1_5260196:@+
batch_normalization_1_5260198:@!
dense_1_5260205:@T
dense_1_5260207:T+
batch_normalization_2_5260210:T+
batch_normalization_2_5260212:T+
batch_normalization_2_5260214:T+
batch_normalization_2_5260216:T
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв'dense_1_dropout/StatefulPartitionedCallв'stream_0_conv_1/StatefulPartitionedCallв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв'stream_0_conv_2/StatefulPartitionedCallв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpв'stream_0_drop_1/StatefulPartitionedCallв'stream_0_drop_2/StatefulPartitionedCallв+stream_0_input_drop/StatefulPartitionedCallЦ
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_52601212-
+stream_0_input_drop/StatefulPartitionedCallЁ
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_5260171stream_0_conv_1_5260173*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_52596302)
'stream_0_conv_1/StatefulPartitionedCall╝
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_5260176batch_normalization_5260178batch_normalization_5260180batch_normalization_5260182*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_52600802-
+batch_normalization/StatefulPartitionedCallС
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_52596702
activation/PartitionedCall╒
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_52600222)
'stream_0_drop_1/StatefulPartitionedCallь
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_drop_1/StatefulPartitionedCall:output:0stream_0_conv_2_5260187stream_0_conv_2_5260189*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_52597002)
'stream_0_conv_2/StatefulPartitionedCall╩
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_5260192batch_normalization_1_5260194batch_normalization_1_5260196batch_normalization_1_5260198*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_52599812/
-batch_normalization_1/StatefulPartitionedCallЩ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_52597402
activation_1/PartitionedCall╙
'stream_0_drop_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0(^stream_0_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_52599232)
'stream_0_drop_2/StatefulPartitionedCall▓
(global_average_pooling1d/PartitionedCallPartitionedCall0stream_0_drop_2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_52597542*
(global_average_pooling1d/PartitionedCall┌
'dense_1_dropout/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0(^stream_0_drop_2/StatefulPartitionedCall*
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
GPU2*0J 8В *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_52598952)
'dense_1_dropout/StatefulPartitionedCall┐
dense_1/StatefulPartitionedCallStatefulPartitionedCall0dense_1_dropout/StatefulPartitionedCall:output:0dense_1_5260205dense_1_5260207*
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
GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_52597792!
dense_1/StatefulPartitionedCall╜
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_5260210batch_normalization_2_5260212batch_normalization_2_5260214batch_normalization_2_5260216*
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
GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_52595172/
-batch_normalization_2/StatefulPartitionedCallж
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_52597992$
"dense_activation_1/PartitionedCall╩
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_5260171*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
&stream_0_conv_1/kernel/Regularizer/mul╨
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_5260187*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulо
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_5260205*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

Identityб
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^dense_1_dropout/StatefulPartitionedCall(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall(^stream_0_drop_2/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ╓: : : : : : : : : : : : : : : : : : 2Z
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
:         ╓
 
_user_specified_nameinputs
Н	
╨
5__inference_batch_normalization_layer_call_fn_5262983

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                   *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_52591092
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                   2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
ї
e
I__inference_activation_1_layer_call_and_return_conditional_losses_5263260

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         ╓@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         ╓@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓@:T P
,
_output_shapes
:         ╓@
 
_user_specified_nameinputs
┐
k
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_5263331

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┴
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seed╖2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
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
╧
M
1__inference_dense_1_dropout_layer_call_fn_5263336

inputs
identity═
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
GPU2*0J 8В *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_52597612
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
═*
ы
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5259517

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
ж
┴
+__inference_basemodel_layer_call_fn_5259859
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
identityИвStatefulPartitionedCall╥
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
:         T*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_52598202
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
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ╓: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:         ╓
"
_user_specified_name
inputs_0
╬
n
5__inference_stream_0_input_drop_layer_call_fn_5262826

inputs
identityИвStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_52601212
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ╓2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ╓
 
_user_specified_nameinputs
╢
п
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5259109

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                   2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                   2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                   2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
Ў
k
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_5260022

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
:         ╓ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ╓ *
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
:         ╓ 2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓ 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ╓ 2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ╓ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓ :T P
,
_output_shapes
:         ╓ 
 
_user_specified_nameinputs
╞
V
*__inference_distance_layer_call_fn_5262793
inputs_0
inputs_1
identity╙
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_distance_layer_call_and_return_conditional_losses_52606472
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         T:         T:Q M
'
_output_shapes
:         T
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         T
"
_user_specified_name
inputs/1
Ў
k
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_5263049

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
:         ╓ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ╓ *
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
:         ╓ 2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓ 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ╓ 2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ╓ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓ :T P
,
_output_shapes
:         ╓ 
 
_user_specified_nameinputs
╗
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5263298

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
а
┴
+__inference_basemodel_layer_call_fn_5260319
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
identityИвStatefulPartitionedCall╠
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
:         T*.
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_52602392
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
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ╓: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:         ╓
"
_user_specified_name
inputs_0
╢
п
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5262882

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                   2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                   2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                   2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
К╖
╡#
B__inference_model_layer_call_and_return_conditional_losses_5261929
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
identityИв-basemodel/batch_normalization/AssignMovingAvgв<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOpв/basemodel/batch_normalization/AssignMovingAvg_1в>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOpв/basemodel/batch_normalization/AssignMovingAvg_2в>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOpв/basemodel/batch_normalization/AssignMovingAvg_3в>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOpв6basemodel/batch_normalization/batchnorm/ReadVariableOpв:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpв8basemodel/batch_normalization/batchnorm_1/ReadVariableOpв<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpв/basemodel/batch_normalization_1/AssignMovingAvgв>basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOpв1basemodel/batch_normalization_1/AssignMovingAvg_1в@basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpв1basemodel/batch_normalization_1/AssignMovingAvg_2в@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOpв1basemodel/batch_normalization_1/AssignMovingAvg_3в@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOpв8basemodel/batch_normalization_1/batchnorm/ReadVariableOpв<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpв:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpв>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpв/basemodel/batch_normalization_2/AssignMovingAvgв>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOpв1basemodel/batch_normalization_2/AssignMovingAvg_1в@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpв1basemodel/batch_normalization_2/AssignMovingAvg_2в@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOpв1basemodel/batch_normalization_2/AssignMovingAvg_3в@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOpв8basemodel/batch_normalization_2/batchnorm/ReadVariableOpв<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpв:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpв>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpв(basemodel/dense_1/BiasAdd/ReadVariableOpв*basemodel/dense_1/BiasAdd_1/ReadVariableOpв'basemodel/dense_1/MatMul/ReadVariableOpв)basemodel/dense_1/MatMul_1/ReadVariableOpв0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpв2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpв<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpв>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpв0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpв2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpв<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpв>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpЯ
+basemodel/stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2-
+basemodel/stream_0_input_drop/dropout/Const╘
)basemodel/stream_0_input_drop/dropout/MulMulinputs_04basemodel/stream_0_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:         ╓2+
)basemodel/stream_0_input_drop/dropout/MulТ
+basemodel/stream_0_input_drop/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:2-
+basemodel/stream_0_input_drop/dropout/Shapeо
Bbasemodel/stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform4basemodel/stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:         ╓*
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
:         ╓24
2basemodel/stream_0_input_drop/dropout/GreaterEqual▐
*basemodel/stream_0_input_drop/dropout/CastCast6basemodel/stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓2,
*basemodel/stream_0_input_drop/dropout/Castў
+basemodel/stream_0_input_drop/dropout/Mul_1Mul-basemodel/stream_0_input_drop/dropout/Mul:z:0.basemodel/stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:         ╓2-
+basemodel/stream_0_input_drop/dropout/Mul_1н
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
:         ╓2-
+basemodel/stream_0_conv_1/conv1d/ExpandDimsЖ
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2/
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1Я
 basemodel/stream_0_conv_1/conv1dConv2D4basemodel/stream_0_conv_1/conv1d/ExpandDims:output:06basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓ *
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1dс
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ╓ *
squeeze_dims

¤        2*
(basemodel/stream_0_conv_1/conv1d/Squeeze┌
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpї
!basemodel/stream_0_conv_1/BiasAddBiasAdd1basemodel/stream_0_conv_1/conv1d/Squeeze:output:08basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓ 2#
!basemodel/stream_0_conv_1/BiasAdd═
<basemodel/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2>
<basemodel/batch_normalization/moments/mean/reduction_indicesС
*basemodel/batch_normalization/moments/meanMean*basemodel/stream_0_conv_1/BiasAdd:output:0Ebasemodel/batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2,
*basemodel/batch_normalization/moments/mean┌
2basemodel/batch_normalization/moments/StopGradientStopGradient3basemodel/batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
: 24
2basemodel/batch_normalization/moments/StopGradientз
7basemodel/batch_normalization/moments/SquaredDifferenceSquaredDifference*basemodel/stream_0_conv_1/BiasAdd:output:0;basemodel/batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:         ╓ 29
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
: *
	keep_dims(20
.basemodel/batch_normalization/moments/variance█
-basemodel/batch_normalization/moments/SqueezeSqueeze3basemodel/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2/
-basemodel/batch_normalization/moments/Squeezeу
/basemodel/batch_normalization/moments/Squeeze_1Squeeze7basemodel/batch_normalization/moments/variance:output:0*
T0*
_output_shapes
: *
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
: *
dtype02>
<basemodel/batch_normalization/AssignMovingAvg/ReadVariableOpР
1basemodel/batch_normalization/AssignMovingAvg/subSubDbasemodel/batch_normalization/AssignMovingAvg/ReadVariableOp:value:06basemodel/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
: 23
1basemodel/batch_normalization/AssignMovingAvg/subЗ
1basemodel/batch_normalization/AssignMovingAvg/mulMul5basemodel/batch_normalization/AssignMovingAvg/sub:z:0<basemodel/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 23
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
: *
dtype02@
>basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOpШ
3basemodel/batch_normalization/AssignMovingAvg_1/subSubFbasemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:08basemodel/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
: 25
3basemodel/batch_normalization/AssignMovingAvg_1/subП
3basemodel/batch_normalization/AssignMovingAvg_1/mulMul7basemodel/batch_normalization/AssignMovingAvg_1/sub:z:0>basemodel/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 25
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
: 2-
+basemodel/batch_normalization/batchnorm/add╜
-basemodel/batch_normalization/batchnorm/RsqrtRsqrt/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-basemodel/batch_normalization/batchnorm/Rsqrt°
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp¤
+basemodel/batch_normalization/batchnorm/mulMul1basemodel/batch_normalization/batchnorm/Rsqrt:y:0Bbasemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+basemodel/batch_normalization/batchnorm/mul∙
-basemodel/batch_normalization/batchnorm/mul_1Mul*basemodel/stream_0_conv_1/BiasAdd:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓ 2/
-basemodel/batch_normalization/batchnorm/mul_1є
-basemodel/batch_normalization/batchnorm/mul_2Mul6basemodel/batch_normalization/moments/Squeeze:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-basemodel/batch_normalization/batchnorm/mul_2ь
6basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6basemodel/batch_normalization/batchnorm/ReadVariableOp∙
+basemodel/batch_normalization/batchnorm/subSub>basemodel/batch_normalization/batchnorm/ReadVariableOp:value:01basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+basemodel/batch_normalization/batchnorm/subВ
-basemodel/batch_normalization/batchnorm/add_1AddV21basemodel/batch_normalization/batchnorm/mul_1:z:0/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ╓ 2/
-basemodel/batch_normalization/batchnorm/add_1и
basemodel/activation/ReluRelu1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ╓ 2
basemodel/activation/ReluЧ
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
:         ╓ 2'
%basemodel/stream_0_drop_1/dropout/Mulй
'basemodel/stream_0_drop_1/dropout/ShapeShape'basemodel/activation/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_1/dropout/Shapeв
>basemodel/stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:         ╓ *
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
:         ╓ 20
.basemodel/stream_0_drop_1/dropout/GreaterEqual╥
&basemodel/stream_0_drop_1/dropout/CastCast2basemodel/stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓ 2(
&basemodel/stream_0_drop_1/dropout/Castч
'basemodel/stream_0_drop_1/dropout/Mul_1Mul)basemodel/stream_0_drop_1/dropout/Mul:z:0*basemodel/stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:         ╓ 2)
'basemodel/stream_0_drop_1/dropout/Mul_1н
/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        21
/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimК
+basemodel/stream_0_conv_2/conv1d/ExpandDims
ExpandDims+basemodel/stream_0_drop_1/dropout/Mul_1:z:08basemodel/stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ╓ 2-
+basemodel/stream_0_conv_2/conv1d/ExpandDimsЖ
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02>
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpи
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
: @2/
-basemodel/stream_0_conv_2/conv1d/ExpandDims_1Я
 basemodel/stream_0_conv_2/conv1dConv2D4basemodel/stream_0_conv_2/conv1d/ExpandDims:output:06basemodel/stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_2/conv1dс
(basemodel/stream_0_conv_2/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:         ╓@*
squeeze_dims

¤        2*
(basemodel/stream_0_conv_2/conv1d/Squeeze┌
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpї
!basemodel/stream_0_conv_2/BiasAddBiasAdd1basemodel/stream_0_conv_2/conv1d/Squeeze:output:08basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓@2#
!basemodel/stream_0_conv_2/BiasAdd╤
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
,basemodel/batch_normalization_1/moments/meanр
4basemodel/batch_normalization_1/moments/StopGradientStopGradient5basemodel/batch_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:@26
4basemodel/batch_normalization_1/moments/StopGradientн
9basemodel/batch_normalization_1/moments/SquaredDifferenceSquaredDifference*basemodel/stream_0_conv_2/BiasAdd:output:0=basemodel/batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:         ╓@2;
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
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_0_conv_2/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓@21
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
:         ╓@21
/basemodel/batch_normalization_1/batchnorm/add_1о
basemodel/activation_1/ReluRelu3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ╓@2
basemodel/activation_1/ReluЧ
'basemodel/stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2)
'basemodel/stream_0_drop_2/dropout/Constщ
%basemodel/stream_0_drop_2/dropout/MulMul)basemodel/activation_1/Relu:activations:00basemodel/stream_0_drop_2/dropout/Const:output:0*
T0*,
_output_shapes
:         ╓@2'
%basemodel/stream_0_drop_2/dropout/Mulл
'basemodel/stream_0_drop_2/dropout/ShapeShape)basemodel/activation_1/Relu:activations:0*
T0*
_output_shapes
:2)
'basemodel/stream_0_drop_2/dropout/Shapeв
>basemodel/stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform0basemodel/stream_0_drop_2/dropout/Shape:output:0*
T0*,
_output_shapes
:         ╓@*
dtype0*
seed╖*
seed2╖2@
>basemodel/stream_0_drop_2/dropout/random_uniform/RandomUniformй
0basemodel/stream_0_drop_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>22
0basemodel/stream_0_drop_2/dropout/GreaterEqual/yл
.basemodel/stream_0_drop_2/dropout/GreaterEqualGreaterEqualGbasemodel/stream_0_drop_2/dropout/random_uniform/RandomUniform:output:09basemodel/stream_0_drop_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ╓@20
.basemodel/stream_0_drop_2/dropout/GreaterEqual╥
&basemodel/stream_0_drop_2/dropout/CastCast2basemodel/stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓@2(
&basemodel/stream_0_drop_2/dropout/Castч
'basemodel/stream_0_drop_2/dropout/Mul_1Mul)basemodel/stream_0_drop_2/dropout/Mul:z:0*basemodel/stream_0_drop_2/dropout/Cast:y:0*
T0*,
_output_shapes
:         ╓@2)
'basemodel/stream_0_drop_2/dropout/Mul_1╕
9basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9basemodel/global_average_pooling1d/Mean/reduction_indices¤
'basemodel/global_average_pooling1d/MeanMean+basemodel/stream_0_drop_2/dropout/Mul_1:z:0Bbasemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2)
'basemodel/global_average_pooling1d/MeanЧ
'basemodel/dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2)
'basemodel/dense_1_dropout/dropout/Constы
%basemodel/dense_1_dropout/dropout/MulMul0basemodel/global_average_pooling1d/Mean:output:00basemodel/dense_1_dropout/dropout/Const:output:0*
T0*'
_output_shapes
:         @2'
%basemodel/dense_1_dropout/dropout/Mul▓
'basemodel/dense_1_dropout/dropout/ShapeShape0basemodel/global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
:2)
'basemodel/dense_1_dropout/dropout/ShapeП
>basemodel/dense_1_dropout/dropout/random_uniform/RandomUniformRandomUniform0basemodel/dense_1_dropout/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seed╖2@
>basemodel/dense_1_dropout/dropout/random_uniform/RandomUniformй
0basemodel/dense_1_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>22
0basemodel/dense_1_dropout/dropout/GreaterEqual/yж
.basemodel/dense_1_dropout/dropout/GreaterEqualGreaterEqualGbasemodel/dense_1_dropout/dropout/random_uniform/RandomUniform:output:09basemodel/dense_1_dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @20
.basemodel/dense_1_dropout/dropout/GreaterEqual═
&basemodel/dense_1_dropout/dropout/CastCast2basemodel/dense_1_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2(
&basemodel/dense_1_dropout/dropout/Castт
'basemodel/dense_1_dropout/dropout/Mul_1Mul)basemodel/dense_1_dropout/dropout/Mul:z:0*basemodel/dense_1_dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:         @2)
'basemodel/dense_1_dropout/dropout/Mul_1├
'basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02)
'basemodel/dense_1/MatMul/ReadVariableOp╬
basemodel/dense_1/MatMulMatMul+basemodel/dense_1_dropout/dropout/Mul_1:z:0/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
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
>basemodel/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2@
>basemodel/batch_normalization_2/moments/mean/reduction_indicesЛ
,basemodel/batch_normalization_2/moments/meanMean"basemodel/dense_1/BiasAdd:output:0Gbasemodel/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2.
,basemodel/batch_normalization_2/moments/mean▄
4basemodel/batch_normalization_2/moments/StopGradientStopGradient5basemodel/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:T26
4basemodel/batch_normalization_2/moments/StopGradientа
9basemodel/batch_normalization_2/moments/SquaredDifferenceSquaredDifference"basemodel/dense_1/BiasAdd:output:0=basemodel/batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:         T2;
9basemodel/batch_normalization_2/moments/SquaredDifference╥
Bbasemodel/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bbasemodel/batch_normalization_2/moments/variance/reduction_indices▓
0basemodel/batch_normalization_2/moments/varianceMean=basemodel/batch_normalization_2/moments/SquaredDifference:z:0Kbasemodel/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(22
0basemodel/batch_normalization_2/moments/varianceр
/basemodel/batch_normalization_2/moments/SqueezeSqueeze5basemodel/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 21
/basemodel/batch_normalization_2/moments/Squeezeш
1basemodel/batch_normalization_2/moments/Squeeze_1Squeeze9basemodel/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 23
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
:T*
dtype02@
>basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOpШ
3basemodel/batch_normalization_2/AssignMovingAvg/subSubFbasemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:08basemodel/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:T25
3basemodel/batch_normalization_2/AssignMovingAvg/subП
3basemodel/batch_normalization_2/AssignMovingAvg/mulMul7basemodel/batch_normalization_2/AssignMovingAvg/sub:z:0>basemodel/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T25
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
:T*
dtype02B
@basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpа
5basemodel/batch_normalization_2/AssignMovingAvg_1/subSubHbasemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0:basemodel/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_2/AssignMovingAvg_1/subЧ
5basemodel/batch_normalization_2/AssignMovingAvg_1/mulMul9basemodel/batch_normalization_2/AssignMovingAvg_1/sub:z:0@basemodel/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T27
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
:T2/
-basemodel/batch_normalization_2/batchnorm/add├
/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_2/batchnorm/Rsqrt■
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpЕ
-basemodel/batch_normalization_2/batchnorm/mulMul3basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_2/batchnorm/mulЄ
/basemodel/batch_normalization_2/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T21
/basemodel/batch_normalization_2/batchnorm/mul_1√
/basemodel/batch_normalization_2/batchnorm/mul_2Mul8basemodel/batch_normalization_2/moments/Squeeze:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_2/batchnorm/mul_2Є
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02:
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpБ
-basemodel/batch_normalization_2/batchnorm/subSub@basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:03basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_2/batchnorm/subЕ
/basemodel/batch_normalization_2/batchnorm/add_1AddV23basemodel/batch_normalization_2/batchnorm/mul_1:z:01basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T21
/basemodel/batch_normalization_2/batchnorm/add_1╛
$basemodel/dense_activation_1/SigmoidSigmoid3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2&
$basemodel/dense_activation_1/Sigmoidг
-basemodel/stream_0_input_drop/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2/
-basemodel/stream_0_input_drop/dropout_1/Const┌
+basemodel/stream_0_input_drop/dropout_1/MulMulinputs_16basemodel/stream_0_input_drop/dropout_1/Const:output:0*
T0*,
_output_shapes
:         ╓2-
+basemodel/stream_0_input_drop/dropout_1/MulЦ
-basemodel/stream_0_input_drop/dropout_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2/
-basemodel/stream_0_input_drop/dropout_1/Shape┤
Dbasemodel/stream_0_input_drop/dropout_1/random_uniform/RandomUniformRandomUniform6basemodel/stream_0_input_drop/dropout_1/Shape:output:0*
T0*,
_output_shapes
:         ╓*
dtype0*
seed╖*
seed2╖2F
Dbasemodel/stream_0_input_drop/dropout_1/random_uniform/RandomUniform╡
6basemodel/stream_0_input_drop/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>28
6basemodel/stream_0_input_drop/dropout_1/GreaterEqual/y├
4basemodel/stream_0_input_drop/dropout_1/GreaterEqualGreaterEqualMbasemodel/stream_0_input_drop/dropout_1/random_uniform/RandomUniform:output:0?basemodel/stream_0_input_drop/dropout_1/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ╓26
4basemodel/stream_0_input_drop/dropout_1/GreaterEqualф
,basemodel/stream_0_input_drop/dropout_1/CastCast8basemodel/stream_0_input_drop/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓2.
,basemodel/stream_0_input_drop/dropout_1/Cast 
-basemodel/stream_0_input_drop/dropout_1/Mul_1Mul/basemodel/stream_0_input_drop/dropout_1/Mul:z:00basemodel/stream_0_input_drop/dropout_1/Cast:y:0*
T0*,
_output_shapes
:         ╓2/
-basemodel/stream_0_input_drop/dropout_1/Mul_1▒
1basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        23
1basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimЦ
-basemodel/stream_0_conv_1/conv1d_1/ExpandDims
ExpandDims1basemodel/stream_0_input_drop/dropout_1/Mul_1:z:0:basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ╓2/
-basemodel/stream_0_conv_1/conv1d_1/ExpandDimsК
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02@
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpм
3basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimз
/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 21
/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1з
"basemodel/stream_0_conv_1/conv1d_1Conv2D6basemodel/stream_0_conv_1/conv1d_1/ExpandDims:output:08basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓ *
paddingSAME*
strides
2$
"basemodel/stream_0_conv_1/conv1d_1ч
*basemodel/stream_0_conv_1/conv1d_1/SqueezeSqueeze+basemodel/stream_0_conv_1/conv1d_1:output:0*
T0*,
_output_shapes
:         ╓ *
squeeze_dims

¤        2,
*basemodel/stream_0_conv_1/conv1d_1/Squeeze▐
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp¤
#basemodel/stream_0_conv_1/BiasAdd_1BiasAdd3basemodel/stream_0_conv_1/conv1d_1/Squeeze:output:0:basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓ 2%
#basemodel/stream_0_conv_1/BiasAdd_1╤
>basemodel/batch_normalization/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>basemodel/batch_normalization/moments_1/mean/reduction_indicesЩ
,basemodel/batch_normalization/moments_1/meanMean,basemodel/stream_0_conv_1/BiasAdd_1:output:0Gbasemodel/batch_normalization/moments_1/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2.
,basemodel/batch_normalization/moments_1/meanр
4basemodel/batch_normalization/moments_1/StopGradientStopGradient5basemodel/batch_normalization/moments_1/mean:output:0*
T0*"
_output_shapes
: 26
4basemodel/batch_normalization/moments_1/StopGradientп
9basemodel/batch_normalization/moments_1/SquaredDifferenceSquaredDifference,basemodel/stream_0_conv_1/BiasAdd_1:output:0=basemodel/batch_normalization/moments_1/StopGradient:output:0*
T0*,
_output_shapes
:         ╓ 2;
9basemodel/batch_normalization/moments_1/SquaredDifference┘
Bbasemodel/batch_normalization/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bbasemodel/batch_normalization/moments_1/variance/reduction_indices╢
0basemodel/batch_normalization/moments_1/varianceMean=basemodel/batch_normalization/moments_1/SquaredDifference:z:0Kbasemodel/batch_normalization/moments_1/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(22
0basemodel/batch_normalization/moments_1/varianceс
/basemodel/batch_normalization/moments_1/SqueezeSqueeze5basemodel/batch_normalization/moments_1/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 21
/basemodel/batch_normalization/moments_1/Squeezeщ
1basemodel/batch_normalization/moments_1/Squeeze_1Squeeze9basemodel/batch_normalization/moments_1/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 23
1basemodel/batch_normalization/moments_1/Squeeze_1│
5basemodel/batch_normalization/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<27
5basemodel/batch_normalization/AssignMovingAvg_2/decay▓
>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_assignmovingavg_readvariableop_resource.^basemodel/batch_normalization/AssignMovingAvg*
_output_shapes
: *
dtype02@
>basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOpШ
3basemodel/batch_normalization/AssignMovingAvg_2/subSubFbasemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp:value:08basemodel/batch_normalization/moments_1/Squeeze:output:0*
T0*
_output_shapes
: 25
3basemodel/batch_normalization/AssignMovingAvg_2/subП
3basemodel/batch_normalization/AssignMovingAvg_2/mulMul7basemodel/batch_normalization/AssignMovingAvg_2/sub:z:0>basemodel/batch_normalization/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes
: 25
3basemodel/batch_normalization/AssignMovingAvg_2/mulН
/basemodel/batch_normalization/AssignMovingAvg_2AssignSubVariableOpEbasemodel_batch_normalization_assignmovingavg_readvariableop_resource7basemodel/batch_normalization/AssignMovingAvg_2/mul:z:0.^basemodel/batch_normalization/AssignMovingAvg?^basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization/AssignMovingAvg_2│
5basemodel/batch_normalization/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<27
5basemodel/batch_normalization/AssignMovingAvg_3/decay╢
>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource0^basemodel/batch_normalization/AssignMovingAvg_1*
_output_shapes
: *
dtype02@
>basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOpЪ
3basemodel/batch_normalization/AssignMovingAvg_3/subSubFbasemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp:value:0:basemodel/batch_normalization/moments_1/Squeeze_1:output:0*
T0*
_output_shapes
: 25
3basemodel/batch_normalization/AssignMovingAvg_3/subП
3basemodel/batch_normalization/AssignMovingAvg_3/mulMul7basemodel/batch_normalization/AssignMovingAvg_3/sub:z:0>basemodel/batch_normalization/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes
: 25
3basemodel/batch_normalization/AssignMovingAvg_3/mulС
/basemodel/batch_normalization/AssignMovingAvg_3AssignSubVariableOpGbasemodel_batch_normalization_assignmovingavg_1_readvariableop_resource7basemodel/batch_normalization/AssignMovingAvg_3/mul:z:00^basemodel/batch_normalization/AssignMovingAvg_1?^basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp*
_output_shapes
 *
dtype021
/basemodel/batch_normalization/AssignMovingAvg_3з
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
: 2/
-basemodel/batch_normalization/batchnorm_1/add├
/basemodel/batch_normalization/batchnorm_1/RsqrtRsqrt1basemodel/batch_normalization/batchnorm_1/add:z:0*
T0*
_output_shapes
: 21
/basemodel/batch_normalization/batchnorm_1/Rsqrt№
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02>
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpЕ
-basemodel/batch_normalization/batchnorm_1/mulMul3basemodel/batch_normalization/batchnorm_1/Rsqrt:y:0Dbasemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2/
-basemodel/batch_normalization/batchnorm_1/mulБ
/basemodel/batch_normalization/batchnorm_1/mul_1Mul,basemodel/stream_0_conv_1/BiasAdd_1:output:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:         ╓ 21
/basemodel/batch_normalization/batchnorm_1/mul_1√
/basemodel/batch_normalization/batchnorm_1/mul_2Mul8basemodel/batch_normalization/moments_1/Squeeze:output:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*
_output_shapes
: 21
/basemodel/batch_normalization/batchnorm_1/mul_2Ё
8basemodel/batch_normalization/batchnorm_1/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02:
8basemodel/batch_normalization/batchnorm_1/ReadVariableOpБ
-basemodel/batch_normalization/batchnorm_1/subSub@basemodel/batch_normalization/batchnorm_1/ReadVariableOp:value:03basemodel/batch_normalization/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
: 2/
-basemodel/batch_normalization/batchnorm_1/subК
/basemodel/batch_normalization/batchnorm_1/add_1AddV23basemodel/batch_normalization/batchnorm_1/mul_1:z:01basemodel/batch_normalization/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:         ╓ 21
/basemodel/batch_normalization/batchnorm_1/add_1о
basemodel/activation/Relu_1Relu3basemodel/batch_normalization/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:         ╓ 2
basemodel/activation/Relu_1Ы
)basemodel/stream_0_drop_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2+
)basemodel/stream_0_drop_1/dropout_1/Constя
'basemodel/stream_0_drop_1/dropout_1/MulMul)basemodel/activation/Relu_1:activations:02basemodel/stream_0_drop_1/dropout_1/Const:output:0*
T0*,
_output_shapes
:         ╓ 2)
'basemodel/stream_0_drop_1/dropout_1/Mulп
)basemodel/stream_0_drop_1/dropout_1/ShapeShape)basemodel/activation/Relu_1:activations:0*
T0*
_output_shapes
:2+
)basemodel/stream_0_drop_1/dropout_1/Shapeи
@basemodel/stream_0_drop_1/dropout_1/random_uniform/RandomUniformRandomUniform2basemodel/stream_0_drop_1/dropout_1/Shape:output:0*
T0*,
_output_shapes
:         ╓ *
dtype0*
seed╖*
seed2╖2B
@basemodel/stream_0_drop_1/dropout_1/random_uniform/RandomUniformн
2basemodel/stream_0_drop_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>24
2basemodel/stream_0_drop_1/dropout_1/GreaterEqual/y│
0basemodel/stream_0_drop_1/dropout_1/GreaterEqualGreaterEqualIbasemodel/stream_0_drop_1/dropout_1/random_uniform/RandomUniform:output:0;basemodel/stream_0_drop_1/dropout_1/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ╓ 22
0basemodel/stream_0_drop_1/dropout_1/GreaterEqual╪
(basemodel/stream_0_drop_1/dropout_1/CastCast4basemodel/stream_0_drop_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓ 2*
(basemodel/stream_0_drop_1/dropout_1/Castя
)basemodel/stream_0_drop_1/dropout_1/Mul_1Mul+basemodel/stream_0_drop_1/dropout_1/Mul:z:0,basemodel/stream_0_drop_1/dropout_1/Cast:y:0*
T0*,
_output_shapes
:         ╓ 2+
)basemodel/stream_0_drop_1/dropout_1/Mul_1▒
1basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        23
1basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dimТ
-basemodel/stream_0_conv_2/conv1d_1/ExpandDims
ExpandDims-basemodel/stream_0_drop_1/dropout_1/Mul_1:z:0:basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ╓ 2/
-basemodel/stream_0_conv_2/conv1d_1/ExpandDimsК
>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02@
>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpм
3basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dimз
/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @21
/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1з
"basemodel/stream_0_conv_2/conv1d_1Conv2D6basemodel/stream_0_conv_2/conv1d_1/ExpandDims:output:08basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓@*
paddingSAME*
strides
2$
"basemodel/stream_0_conv_2/conv1d_1ч
*basemodel/stream_0_conv_2/conv1d_1/SqueezeSqueeze+basemodel/stream_0_conv_2/conv1d_1:output:0*
T0*,
_output_shapes
:         ╓@*
squeeze_dims

¤        2,
*basemodel/stream_0_conv_2/conv1d_1/Squeeze▐
2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp¤
#basemodel/stream_0_conv_2/BiasAdd_1BiasAdd3basemodel/stream_0_conv_2/conv1d_1/Squeeze:output:0:basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓@2%
#basemodel/stream_0_conv_2/BiasAdd_1╒
@basemodel/batch_normalization_1/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2B
@basemodel/batch_normalization_1/moments_1/mean/reduction_indicesЯ
.basemodel/batch_normalization_1/moments_1/meanMean,basemodel/stream_0_conv_2/BiasAdd_1:output:0Ibasemodel/batch_normalization_1/moments_1/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(20
.basemodel/batch_normalization_1/moments_1/meanц
6basemodel/batch_normalization_1/moments_1/StopGradientStopGradient7basemodel/batch_normalization_1/moments_1/mean:output:0*
T0*"
_output_shapes
:@28
6basemodel/batch_normalization_1/moments_1/StopGradient╡
;basemodel/batch_normalization_1/moments_1/SquaredDifferenceSquaredDifference,basemodel/stream_0_conv_2/BiasAdd_1:output:0?basemodel/batch_normalization_1/moments_1/StopGradient:output:0*
T0*,
_output_shapes
:         ╓@2=
;basemodel/batch_normalization_1/moments_1/SquaredDifference▌
Dbasemodel/batch_normalization_1/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2F
Dbasemodel/batch_normalization_1/moments_1/variance/reduction_indices╛
2basemodel/batch_normalization_1/moments_1/varianceMean?basemodel/batch_normalization_1/moments_1/SquaredDifference:z:0Mbasemodel/batch_normalization_1/moments_1/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(24
2basemodel/batch_normalization_1/moments_1/varianceч
1basemodel/batch_normalization_1/moments_1/SqueezeSqueeze7basemodel/batch_normalization_1/moments_1/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 23
1basemodel/batch_normalization_1/moments_1/Squeezeя
3basemodel/batch_normalization_1/moments_1/Squeeze_1Squeeze;basemodel/batch_normalization_1/moments_1/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 25
3basemodel/batch_normalization_1/moments_1/Squeeze_1╖
7basemodel/batch_normalization_1/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<29
7basemodel/batch_normalization_1/AssignMovingAvg_2/decay║
@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_1_assignmovingavg_readvariableop_resource0^basemodel/batch_normalization_1/AssignMovingAvg*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOpа
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
1basemodel/batch_normalization_1/AssignMovingAvg_2╖
7basemodel/batch_normalization_1/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<29
7basemodel/batch_normalization_1/AssignMovingAvg_3/decay╛
@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource2^basemodel/batch_normalization_1/AssignMovingAvg_1*
_output_shapes
:@*
dtype02B
@basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOpв
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
1basemodel/batch_normalization_1/AssignMovingAvg_3л
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
/basemodel/batch_normalization_1/batchnorm_1/add╔
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
/basemodel/batch_normalization_1/batchnorm_1/mulЗ
1basemodel/batch_normalization_1/batchnorm_1/mul_1Mul,basemodel/stream_0_conv_2/BiasAdd_1:output:03basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:         ╓@23
1basemodel/batch_normalization_1/batchnorm_1/mul_1Г
1basemodel/batch_normalization_1/batchnorm_1/mul_2Mul:basemodel/batch_normalization_1/moments_1/Squeeze:output:03basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization_1/batchnorm_1/mul_2Ў
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpЙ
/basemodel/batch_normalization_1/batchnorm_1/subSubBbasemodel/batch_normalization_1/batchnorm_1/ReadVariableOp:value:05basemodel/batch_normalization_1/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm_1/subТ
1basemodel/batch_normalization_1/batchnorm_1/add_1AddV25basemodel/batch_normalization_1/batchnorm_1/mul_1:z:03basemodel/batch_normalization_1/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:         ╓@23
1basemodel/batch_normalization_1/batchnorm_1/add_1┤
basemodel/activation_1/Relu_1Relu5basemodel/batch_normalization_1/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:         ╓@2
basemodel/activation_1/Relu_1Ы
)basemodel/stream_0_drop_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2+
)basemodel/stream_0_drop_2/dropout_1/Constё
'basemodel/stream_0_drop_2/dropout_1/MulMul+basemodel/activation_1/Relu_1:activations:02basemodel/stream_0_drop_2/dropout_1/Const:output:0*
T0*,
_output_shapes
:         ╓@2)
'basemodel/stream_0_drop_2/dropout_1/Mul▒
)basemodel/stream_0_drop_2/dropout_1/ShapeShape+basemodel/activation_1/Relu_1:activations:0*
T0*
_output_shapes
:2+
)basemodel/stream_0_drop_2/dropout_1/Shapeи
@basemodel/stream_0_drop_2/dropout_1/random_uniform/RandomUniformRandomUniform2basemodel/stream_0_drop_2/dropout_1/Shape:output:0*
T0*,
_output_shapes
:         ╓@*
dtype0*
seed╖*
seed2╖2B
@basemodel/stream_0_drop_2/dropout_1/random_uniform/RandomUniformн
2basemodel/stream_0_drop_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>24
2basemodel/stream_0_drop_2/dropout_1/GreaterEqual/y│
0basemodel/stream_0_drop_2/dropout_1/GreaterEqualGreaterEqualIbasemodel/stream_0_drop_2/dropout_1/random_uniform/RandomUniform:output:0;basemodel/stream_0_drop_2/dropout_1/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ╓@22
0basemodel/stream_0_drop_2/dropout_1/GreaterEqual╪
(basemodel/stream_0_drop_2/dropout_1/CastCast4basemodel/stream_0_drop_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓@2*
(basemodel/stream_0_drop_2/dropout_1/Castя
)basemodel/stream_0_drop_2/dropout_1/Mul_1Mul+basemodel/stream_0_drop_2/dropout_1/Mul:z:0,basemodel/stream_0_drop_2/dropout_1/Cast:y:0*
T0*,
_output_shapes
:         ╓@2+
)basemodel/stream_0_drop_2/dropout_1/Mul_1╝
;basemodel/global_average_pooling1d/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d/Mean_1/reduction_indicesЕ
)basemodel/global_average_pooling1d/Mean_1Mean-basemodel/stream_0_drop_2/dropout_1/Mul_1:z:0Dbasemodel/global_average_pooling1d/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:         @2+
)basemodel/global_average_pooling1d/Mean_1Ы
)basemodel/dense_1_dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2+
)basemodel/dense_1_dropout/dropout_1/Constє
'basemodel/dense_1_dropout/dropout_1/MulMul2basemodel/global_average_pooling1d/Mean_1:output:02basemodel/dense_1_dropout/dropout_1/Const:output:0*
T0*'
_output_shapes
:         @2)
'basemodel/dense_1_dropout/dropout_1/Mul╕
)basemodel/dense_1_dropout/dropout_1/ShapeShape2basemodel/global_average_pooling1d/Mean_1:output:0*
T0*
_output_shapes
:2+
)basemodel/dense_1_dropout/dropout_1/Shapeв
@basemodel/dense_1_dropout/dropout_1/random_uniform/RandomUniformRandomUniform2basemodel/dense_1_dropout/dropout_1/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seed╖*
seed22B
@basemodel/dense_1_dropout/dropout_1/random_uniform/RandomUniformн
2basemodel/dense_1_dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>24
2basemodel/dense_1_dropout/dropout_1/GreaterEqual/yо
0basemodel/dense_1_dropout/dropout_1/GreaterEqualGreaterEqualIbasemodel/dense_1_dropout/dropout_1/random_uniform/RandomUniform:output:0;basemodel/dense_1_dropout/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @22
0basemodel/dense_1_dropout/dropout_1/GreaterEqual╙
(basemodel/dense_1_dropout/dropout_1/CastCast4basemodel/dense_1_dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2*
(basemodel/dense_1_dropout/dropout_1/Castъ
)basemodel/dense_1_dropout/dropout_1/Mul_1Mul+basemodel/dense_1_dropout/dropout_1/Mul:z:0,basemodel/dense_1_dropout/dropout_1/Cast:y:0*
T0*'
_output_shapes
:         @2+
)basemodel/dense_1_dropout/dropout_1/Mul_1╟
)basemodel/dense_1/MatMul_1/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02+
)basemodel/dense_1/MatMul_1/ReadVariableOp╓
basemodel/dense_1/MatMul_1MatMul-basemodel/dense_1_dropout/dropout_1/Mul_1:z:01basemodel/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
basemodel/dense_1/MatMul_1╞
*basemodel/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02,
*basemodel/dense_1/BiasAdd_1/ReadVariableOp╤
basemodel/dense_1/BiasAdd_1BiasAdd$basemodel/dense_1/MatMul_1:product:02basemodel/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
basemodel/dense_1/BiasAdd_1╬
@basemodel/batch_normalization_2/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2B
@basemodel/batch_normalization_2/moments_1/mean/reduction_indicesУ
.basemodel/batch_normalization_2/moments_1/meanMean$basemodel/dense_1/BiasAdd_1:output:0Ibasemodel/batch_normalization_2/moments_1/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(20
.basemodel/batch_normalization_2/moments_1/meanт
6basemodel/batch_normalization_2/moments_1/StopGradientStopGradient7basemodel/batch_normalization_2/moments_1/mean:output:0*
T0*
_output_shapes

:T28
6basemodel/batch_normalization_2/moments_1/StopGradientи
;basemodel/batch_normalization_2/moments_1/SquaredDifferenceSquaredDifference$basemodel/dense_1/BiasAdd_1:output:0?basemodel/batch_normalization_2/moments_1/StopGradient:output:0*
T0*'
_output_shapes
:         T2=
;basemodel/batch_normalization_2/moments_1/SquaredDifference╓
Dbasemodel/batch_normalization_2/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dbasemodel/batch_normalization_2/moments_1/variance/reduction_indices║
2basemodel/batch_normalization_2/moments_1/varianceMean?basemodel/batch_normalization_2/moments_1/SquaredDifference:z:0Mbasemodel/batch_normalization_2/moments_1/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(24
2basemodel/batch_normalization_2/moments_1/varianceц
1basemodel/batch_normalization_2/moments_1/SqueezeSqueeze7basemodel/batch_normalization_2/moments_1/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 23
1basemodel/batch_normalization_2/moments_1/Squeezeю
3basemodel/batch_normalization_2/moments_1/Squeeze_1Squeeze;basemodel/batch_normalization_2/moments_1/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 25
3basemodel/batch_normalization_2/moments_1/Squeeze_1╖
7basemodel/batch_normalization_2/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<29
7basemodel/batch_normalization_2/AssignMovingAvg_2/decay║
@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOpReadVariableOpGbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource0^basemodel/batch_normalization_2/AssignMovingAvg*
_output_shapes
:T*
dtype02B
@basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOpа
5basemodel/batch_normalization_2/AssignMovingAvg_2/subSubHbasemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp:value:0:basemodel/batch_normalization_2/moments_1/Squeeze:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_2/AssignMovingAvg_2/subЧ
5basemodel/batch_normalization_2/AssignMovingAvg_2/mulMul9basemodel/batch_normalization_2/AssignMovingAvg_2/sub:z:0@basemodel/batch_normalization_2/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_2/AssignMovingAvg_2/mulЩ
1basemodel/batch_normalization_2/AssignMovingAvg_2AssignSubVariableOpGbasemodel_batch_normalization_2_assignmovingavg_readvariableop_resource9basemodel/batch_normalization_2/AssignMovingAvg_2/mul:z:00^basemodel/batch_normalization_2/AssignMovingAvgA^basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_2/AssignMovingAvg_2╖
7basemodel/batch_normalization_2/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<29
7basemodel/batch_normalization_2/AssignMovingAvg_3/decay╛
@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOpReadVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource2^basemodel/batch_normalization_2/AssignMovingAvg_1*
_output_shapes
:T*
dtype02B
@basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOpв
5basemodel/batch_normalization_2/AssignMovingAvg_3/subSubHbasemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp:value:0<basemodel/batch_normalization_2/moments_1/Squeeze_1:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_2/AssignMovingAvg_3/subЧ
5basemodel/batch_normalization_2/AssignMovingAvg_3/mulMul9basemodel/batch_normalization_2/AssignMovingAvg_3/sub:z:0@basemodel/batch_normalization_2/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes
:T27
5basemodel/batch_normalization_2/AssignMovingAvg_3/mulЭ
1basemodel/batch_normalization_2/AssignMovingAvg_3AssignSubVariableOpIbasemodel_batch_normalization_2_assignmovingavg_1_readvariableop_resource9basemodel/batch_normalization_2/AssignMovingAvg_3/mul:z:02^basemodel/batch_normalization_2/AssignMovingAvg_1A^basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp*
_output_shapes
 *
dtype023
1basemodel/batch_normalization_2/AssignMovingAvg_3л
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
:T21
/basemodel/batch_normalization_2/batchnorm_1/add╔
1basemodel/batch_normalization_2/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_2/batchnorm_1/add:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_2/batchnorm_1/RsqrtВ
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02@
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpН
/basemodel/batch_normalization_2/batchnorm_1/mulMul5basemodel/batch_normalization_2/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_2/batchnorm_1/mul·
1basemodel/batch_normalization_2/batchnorm_1/mul_1Mul$basemodel/dense_1/BiasAdd_1:output:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*'
_output_shapes
:         T23
1basemodel/batch_normalization_2/batchnorm_1/mul_1Г
1basemodel/batch_normalization_2/batchnorm_1/mul_2Mul:basemodel/batch_normalization_2/moments_1/Squeeze:output:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_2/batchnorm_1/mul_2Ў
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpЙ
/basemodel/batch_normalization_2/batchnorm_1/subSubBbasemodel/batch_normalization_2/batchnorm_1/ReadVariableOp:value:05basemodel/batch_normalization_2/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_2/batchnorm_1/subН
1basemodel/batch_normalization_2/batchnorm_1/add_1AddV25basemodel/batch_normalization_2/batchnorm_1/mul_1:z:03basemodel/batch_normalization_2/batchnorm_1/sub:z:0*
T0*'
_output_shapes
:         T23
1basemodel/batch_normalization_2/batchnorm_1/add_1─
&basemodel/dense_activation_1/Sigmoid_1Sigmoid5basemodel/batch_normalization_2/batchnorm_1/add_1:z:0*
T0*'
_output_shapes
:         T2(
&basemodel/dense_activation_1/Sigmoid_1л
distance/subSub(basemodel/dense_activation_1/Sigmoid:y:0*basemodel/dense_activation_1/Sigmoid_1:y:0*
T0*'
_output_shapes
:         T2
distance/subp
distance/SquareSquaredistance/sub:z:0*
T0*'
_output_shapes
:         T2
distance/SquareЛ
distance/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2 
distance/Sum/reduction_indicesд
distance/SumSumdistance/Square:y:0'distance/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
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
:         2
distance/Maximumn
distance/SqrtSqrtdistance/Maximum:z:0*
T0*'
_output_shapes
:         2
distance/Sqrt°
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul╧
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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
IdentityIdentitydistance/Sqrt:y:0^NoOp*
T0*'
_output_shapes
:         2

Identity╟
NoOpNoOp.^basemodel/batch_normalization/AssignMovingAvg=^basemodel/batch_normalization/AssignMovingAvg/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_1?^basemodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_2?^basemodel/batch_normalization/AssignMovingAvg_2/ReadVariableOp0^basemodel/batch_normalization/AssignMovingAvg_3?^basemodel/batch_normalization/AssignMovingAvg_3/ReadVariableOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization/batchnorm_1/ReadVariableOp=^basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp0^basemodel/batch_normalization_1/AssignMovingAvg?^basemodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_1A^basemodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_2A^basemodel/batch_normalization_1/AssignMovingAvg_2/ReadVariableOp2^basemodel/batch_normalization_1/AssignMovingAvg_3A^basemodel/batch_normalization_1/AssignMovingAvg_3/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp?^basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp0^basemodel/batch_normalization_2/AssignMovingAvg?^basemodel/batch_normalization_2/AssignMovingAvg/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_1A^basemodel/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_2A^basemodel/batch_normalization_2/AssignMovingAvg_2/ReadVariableOp2^basemodel/batch_normalization_2/AssignMovingAvg_3A^basemodel/batch_normalization_2/AssignMovingAvg_3/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp?^basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp+^basemodel/dense_1/BiasAdd_1/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp*^basemodel/dense_1/MatMul_1/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp3^basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp1^basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp3^basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp=^basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:         ╓:         ╓: : : : : : : : : : : : : : : : : : 2^
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
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp2T
(basemodel/dense_1/BiasAdd/ReadVariableOp(basemodel/dense_1/BiasAdd/ReadVariableOp2X
*basemodel/dense_1/BiasAdd_1/ReadVariableOp*basemodel/dense_1/BiasAdd_1/ReadVariableOp2R
'basemodel/dense_1/MatMul/ReadVariableOp'basemodel/dense_1/MatMul/ReadVariableOp2V
)basemodel/dense_1/MatMul_1/ReadVariableOp)basemodel/dense_1/MatMul_1/ReadVariableOp2d
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2h
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2А
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp2h
2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2А
>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:V R
,
_output_shapes
:         ╓
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:         ╓
"
_user_specified_name
inputs/1
╖+
щ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5259169

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
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
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient▒
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                   2
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
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
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
╫#<2
AssignMovingAvg/decayд
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
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
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
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
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                   2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                   2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                   2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
є
c
G__inference_activation_layer_call_and_return_conditional_losses_5259670

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         ╓ 2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         ╓ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓ :T P
,
_output_shapes
:         ╓ 
 
_user_specified_nameinputs
ою
З
"__inference__wrapped_model_5259085
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
identityИв<model/basemodel/batch_normalization/batchnorm/ReadVariableOpв>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1в>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2в@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpв>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOpв@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1в@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2вBmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpв>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOpв@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1в@model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2вBmodel/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpв@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpвBmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1вBmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2вDmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpв>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOpв@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1в@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2вBmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpв@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpвBmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1вBmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2вDmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpв.model/basemodel/dense_1/BiasAdd/ReadVariableOpв0model/basemodel/dense_1/BiasAdd_1/ReadVariableOpв-model/basemodel/dense_1/MatMul/ReadVariableOpв/model/basemodel/dense_1/MatMul_1/ReadVariableOpв6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpв8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpвBmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpвDmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpв6model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpв8model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpвBmodel/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpвDmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpм
,model/basemodel/stream_0_input_drop/IdentityIdentityleft_inputs*
T0*,
_output_shapes
:         ╓2.
,model/basemodel/stream_0_input_drop/Identity╣
5model/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        27
5model/basemodel/stream_0_conv_1/conv1d/ExpandDims/dimж
1model/basemodel/stream_0_conv_1/conv1d/ExpandDims
ExpandDims5model/basemodel/stream_0_input_drop/Identity:output:0>model/basemodel/stream_0_conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ╓23
1model/basemodel/stream_0_conv_1/conv1d/ExpandDimsШ
Bmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02D
Bmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp┤
7model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim╖
3model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1
ExpandDimsJmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0@model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 25
3model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1╖
&model/basemodel/stream_0_conv_1/conv1dConv2D:model/basemodel/stream_0_conv_1/conv1d/ExpandDims:output:0<model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓ *
paddingSAME*
strides
2(
&model/basemodel/stream_0_conv_1/conv1dє
.model/basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze/model/basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ╓ *
squeeze_dims

¤        20
.model/basemodel/stream_0_conv_1/conv1d/Squeezeь
6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp?model_basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpН
'model/basemodel/stream_0_conv_1/BiasAddBiasAdd7model/basemodel/stream_0_conv_1/conv1d/Squeeze:output:0>model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓ 2)
'model/basemodel/stream_0_conv_1/BiasAdd■
<model/basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOpEmodel_basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02>
<model/basemodel/batch_normalization/batchnorm/ReadVariableOpп
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
: 23
1model/basemodel/batch_normalization/batchnorm/add╧
3model/basemodel/batch_normalization/batchnorm/RsqrtRsqrt5model/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 25
3model/basemodel/batch_normalization/batchnorm/RsqrtК
@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpImodel_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02B
@model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOpХ
1model/basemodel/batch_normalization/batchnorm/mulMul7model/basemodel/batch_normalization/batchnorm/Rsqrt:y:0Hmodel/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 23
1model/basemodel/batch_normalization/batchnorm/mulС
3model/basemodel/batch_normalization/batchnorm/mul_1Mul0model/basemodel/stream_0_conv_1/BiasAdd:output:05model/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓ 25
3model/basemodel/batch_normalization/batchnorm/mul_1Д
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02@
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1Х
3model/basemodel/batch_normalization/batchnorm/mul_2MulFmodel/basemodel/batch_normalization/batchnorm/ReadVariableOp_1:value:05model/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 25
3model/basemodel/batch_normalization/batchnorm/mul_2Д
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02@
>model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2У
1model/basemodel/batch_normalization/batchnorm/subSubFmodel/basemodel/batch_normalization/batchnorm/ReadVariableOp_2:value:07model/basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 23
1model/basemodel/batch_normalization/batchnorm/subЪ
3model/basemodel/batch_normalization/batchnorm/add_1AddV27model/basemodel/batch_normalization/batchnorm/mul_1:z:05model/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ╓ 25
3model/basemodel/batch_normalization/batchnorm/add_1║
model/basemodel/activation/ReluRelu7model/basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ╓ 2!
model/basemodel/activation/Relu╞
(model/basemodel/stream_0_drop_1/IdentityIdentity-model/basemodel/activation/Relu:activations:0*
T0*,
_output_shapes
:         ╓ 2*
(model/basemodel/stream_0_drop_1/Identity╣
5model/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        27
5model/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimв
1model/basemodel/stream_0_conv_2/conv1d/ExpandDims
ExpandDims1model/basemodel/stream_0_drop_1/Identity:output:0>model/basemodel/stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ╓ 23
1model/basemodel/stream_0_conv_2/conv1d/ExpandDimsШ
Bmodel/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02D
Bmodel/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp┤
7model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dim╖
3model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1
ExpandDimsJmodel/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:0@model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @25
3model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1╖
&model/basemodel/stream_0_conv_2/conv1dConv2D:model/basemodel/stream_0_conv_2/conv1d/ExpandDims:output:0<model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓@*
paddingSAME*
strides
2(
&model/basemodel/stream_0_conv_2/conv1dє
.model/basemodel/stream_0_conv_2/conv1d/SqueezeSqueeze/model/basemodel/stream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:         ╓@*
squeeze_dims

¤        20
.model/basemodel/stream_0_conv_2/conv1d/Squeezeь
6model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp?model_basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpН
'model/basemodel/stream_0_conv_2/BiasAddBiasAdd7model/basemodel/stream_0_conv_2/conv1d/Squeeze:output:0>model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓@2)
'model/basemodel/stream_0_conv_2/BiasAddД
>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02@
>model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp│
5model/basemodel/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:27
5model/basemodel/batch_normalization_1/batchnorm/add/yа
3model/basemodel/batch_normalization_1/batchnorm/addAddV2Fmodel/basemodel/batch_normalization_1/batchnorm/ReadVariableOp:value:0>model/basemodel/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@25
3model/basemodel/batch_normalization_1/batchnorm/add╒
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
3model/basemodel/batch_normalization_1/batchnorm/mulЧ
5model/basemodel/batch_normalization_1/batchnorm/mul_1Mul0model/basemodel/stream_0_conv_2/BiasAdd:output:07model/basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓@27
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
3model/basemodel/batch_normalization_1/batchnorm/subв
5model/basemodel/batch_normalization_1/batchnorm/add_1AddV29model/basemodel/batch_normalization_1/batchnorm/mul_1:z:07model/basemodel/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ╓@27
5model/basemodel/batch_normalization_1/batchnorm/add_1└
!model/basemodel/activation_1/ReluRelu9model/basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ╓@2#
!model/basemodel/activation_1/Relu╚
(model/basemodel/stream_0_drop_2/IdentityIdentity/model/basemodel/activation_1/Relu:activations:0*
T0*,
_output_shapes
:         ╓@2*
(model/basemodel/stream_0_drop_2/Identity─
?model/basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2A
?model/basemodel/global_average_pooling1d/Mean/reduction_indicesХ
-model/basemodel/global_average_pooling1d/MeanMean1model/basemodel/stream_0_drop_2/Identity:output:0Hmodel/basemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2/
-model/basemodel/global_average_pooling1d/Mean╩
(model/basemodel/dense_1_dropout/IdentityIdentity6model/basemodel/global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:         @2*
(model/basemodel/dense_1_dropout/Identity╒
-model/basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp6model_basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-model/basemodel/dense_1/MatMul/ReadVariableOpц
model/basemodel/dense_1/MatMulMatMul1model/basemodel/dense_1_dropout/Identity:output:05model/basemodel/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2 
model/basemodel/dense_1/MatMul╘
.model/basemodel/dense_1/BiasAdd/ReadVariableOpReadVariableOp7model_basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype020
.model/basemodel/dense_1/BiasAdd/ReadVariableOpс
model/basemodel/dense_1/BiasAddBiasAdd(model/basemodel/dense_1/MatMul:product:06model/basemodel/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2!
model/basemodel/dense_1/BiasAddД
>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02@
>model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp│
5model/basemodel/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:27
5model/basemodel/batch_normalization_2/batchnorm/add/yа
3model/basemodel/batch_normalization_2/batchnorm/addAddV2Fmodel/basemodel/batch_normalization_2/batchnorm/ReadVariableOp:value:0>model/basemodel/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:T25
3model/basemodel/batch_normalization_2/batchnorm/add╒
5model/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt7model/basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_2/batchnorm/RsqrtР
Bmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02D
Bmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpЭ
3model/basemodel/batch_normalization_2/batchnorm/mulMul9model/basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Jmodel/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T25
3model/basemodel/batch_normalization_2/batchnorm/mulК
5model/basemodel/batch_normalization_2/batchnorm/mul_1Mul(model/basemodel/dense_1/BiasAdd:output:07model/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T27
5model/basemodel/batch_normalization_2/batchnorm/mul_1К
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02B
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1Э
5model/basemodel/batch_normalization_2/batchnorm/mul_2MulHmodel/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:value:07model/basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_2/batchnorm/mul_2К
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02B
@model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2Ы
3model/basemodel/batch_normalization_2/batchnorm/subSubHmodel/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:value:09model/basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T25
3model/basemodel/batch_normalization_2/batchnorm/subЭ
5model/basemodel/batch_normalization_2/batchnorm/add_1AddV29model/basemodel/batch_normalization_2/batchnorm/mul_1:z:07model/basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T27
5model/basemodel/batch_normalization_2/batchnorm/add_1╨
*model/basemodel/dense_activation_1/SigmoidSigmoid9model/basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2,
*model/basemodel/dense_activation_1/Sigmoid▒
.model/basemodel/stream_0_input_drop/Identity_1Identityright_inputs*
T0*,
_output_shapes
:         ╓20
.model/basemodel/stream_0_input_drop/Identity_1╜
7model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        29
7model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimо
3model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims
ExpandDims7model/basemodel/stream_0_input_drop/Identity_1:output:0@model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ╓25
3model/basemodel/stream_0_conv_1/conv1d_1/ExpandDimsЬ
Dmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02F
Dmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp╕
9model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dim┐
5model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1
ExpandDimsLmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0Bmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 27
5model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1┐
(model/basemodel/stream_0_conv_1/conv1d_1Conv2D<model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims:output:0>model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓ *
paddingSAME*
strides
2*
(model/basemodel/stream_0_conv_1/conv1d_1∙
0model/basemodel/stream_0_conv_1/conv1d_1/SqueezeSqueeze1model/basemodel/stream_0_conv_1/conv1d_1:output:0*
T0*,
_output_shapes
:         ╓ *
squeeze_dims

¤        22
0model/basemodel/stream_0_conv_1/conv1d_1/SqueezeЁ
8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp?model_basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpХ
)model/basemodel/stream_0_conv_1/BiasAdd_1BiasAdd9model/basemodel/stream_0_conv_1/conv1d_1/Squeeze:output:0@model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓ 2+
)model/basemodel/stream_0_conv_1/BiasAdd_1В
>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOpReadVariableOpEmodel_basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02@
>model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp│
5model/basemodel/batch_normalization/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:27
5model/basemodel/batch_normalization/batchnorm_1/add/yа
3model/basemodel/batch_normalization/batchnorm_1/addAddV2Fmodel/basemodel/batch_normalization/batchnorm_1/ReadVariableOp:value:0>model/basemodel/batch_normalization/batchnorm_1/add/y:output:0*
T0*
_output_shapes
: 25
3model/basemodel/batch_normalization/batchnorm_1/add╒
5model/basemodel/batch_normalization/batchnorm_1/RsqrtRsqrt7model/basemodel/batch_normalization/batchnorm_1/add:z:0*
T0*
_output_shapes
: 27
5model/basemodel/batch_normalization/batchnorm_1/RsqrtО
Bmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpReadVariableOpImodel_basemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02D
Bmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpЭ
3model/basemodel/batch_normalization/batchnorm_1/mulMul9model/basemodel/batch_normalization/batchnorm_1/Rsqrt:y:0Jmodel/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 25
3model/basemodel/batch_normalization/batchnorm_1/mulЩ
5model/basemodel/batch_normalization/batchnorm_1/mul_1Mul2model/basemodel/stream_0_conv_1/BiasAdd_1:output:07model/basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:         ╓ 27
5model/basemodel/batch_normalization/batchnorm_1/mul_1И
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1Э
5model/basemodel/batch_normalization/batchnorm_1/mul_2MulHmodel/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1:value:07model/basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*
_output_shapes
: 27
5model/basemodel/batch_normalization/batchnorm_1/mul_2И
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2ReadVariableOpGmodel_basemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02B
@model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2Ы
3model/basemodel/batch_normalization/batchnorm_1/subSubHmodel/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2:value:09model/basemodel/batch_normalization/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
: 25
3model/basemodel/batch_normalization/batchnorm_1/subв
5model/basemodel/batch_normalization/batchnorm_1/add_1AddV29model/basemodel/batch_normalization/batchnorm_1/mul_1:z:07model/basemodel/batch_normalization/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:         ╓ 27
5model/basemodel/batch_normalization/batchnorm_1/add_1└
!model/basemodel/activation/Relu_1Relu9model/basemodel/batch_normalization/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:         ╓ 2#
!model/basemodel/activation/Relu_1╠
*model/basemodel/stream_0_drop_1/Identity_1Identity/model/basemodel/activation/Relu_1:activations:0*
T0*,
_output_shapes
:         ╓ 2,
*model/basemodel/stream_0_drop_1/Identity_1╜
7model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        29
7model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dimк
3model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims
ExpandDims3model/basemodel/stream_0_drop_1/Identity_1:output:0@model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ╓ 25
3model/basemodel/stream_0_conv_2/conv1d_1/ExpandDimsЬ
Dmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpKmodel_basemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02F
Dmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp╕
9model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dim┐
5model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1
ExpandDimsLmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp:value:0Bmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @27
5model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1┐
(model/basemodel/stream_0_conv_2/conv1d_1Conv2D<model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims:output:0>model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓@*
paddingSAME*
strides
2*
(model/basemodel/stream_0_conv_2/conv1d_1∙
0model/basemodel/stream_0_conv_2/conv1d_1/SqueezeSqueeze1model/basemodel/stream_0_conv_2/conv1d_1:output:0*
T0*,
_output_shapes
:         ╓@*
squeeze_dims

¤        22
0model/basemodel/stream_0_conv_2/conv1d_1/SqueezeЁ
8model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpReadVariableOp?model_basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpХ
)model/basemodel/stream_0_conv_2/BiasAdd_1BiasAdd9model/basemodel/stream_0_conv_2/conv1d_1/Squeeze:output:0@model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓@2+
)model/basemodel/stream_0_conv_2/BiasAdd_1И
@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02B
@model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp╖
7model/basemodel/batch_normalization_1/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:29
7model/basemodel/batch_normalization_1/batchnorm_1/add/yи
5model/basemodel/batch_normalization_1/batchnorm_1/addAddV2Hmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp:value:0@model/basemodel/batch_normalization_1/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_1/batchnorm_1/add█
7model/basemodel/batch_normalization_1/batchnorm_1/RsqrtRsqrt9model/basemodel/batch_normalization_1/batchnorm_1/add:z:0*
T0*
_output_shapes
:@29
7model/basemodel/batch_normalization_1/batchnorm_1/RsqrtФ
Dmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpе
5model/basemodel/batch_normalization_1/batchnorm_1/mulMul;model/basemodel/batch_normalization_1/batchnorm_1/Rsqrt:y:0Lmodel/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_1/batchnorm_1/mulЯ
7model/basemodel/batch_normalization_1/batchnorm_1/mul_1Mul2model/basemodel/stream_0_conv_2/BiasAdd_1:output:09model/basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:         ╓@29
7model/basemodel/batch_normalization_1/batchnorm_1/mul_1О
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1е
7model/basemodel/batch_normalization_1/batchnorm_1/mul_2MulJmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1:value:09model/basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@29
7model/basemodel/batch_normalization_1/batchnorm_1/mul_2О
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02D
Bmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2г
5model/basemodel/batch_normalization_1/batchnorm_1/subSubJmodel/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2:value:0;model/basemodel/batch_normalization_1/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@27
5model/basemodel/batch_normalization_1/batchnorm_1/subк
7model/basemodel/batch_normalization_1/batchnorm_1/add_1AddV2;model/basemodel/batch_normalization_1/batchnorm_1/mul_1:z:09model/basemodel/batch_normalization_1/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:         ╓@29
7model/basemodel/batch_normalization_1/batchnorm_1/add_1╞
#model/basemodel/activation_1/Relu_1Relu;model/basemodel/batch_normalization_1/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:         ╓@2%
#model/basemodel/activation_1/Relu_1╬
*model/basemodel/stream_0_drop_2/Identity_1Identity1model/basemodel/activation_1/Relu_1:activations:0*
T0*,
_output_shapes
:         ╓@2,
*model/basemodel/stream_0_drop_2/Identity_1╚
Amodel/basemodel/global_average_pooling1d/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Amodel/basemodel/global_average_pooling1d/Mean_1/reduction_indicesЭ
/model/basemodel/global_average_pooling1d/Mean_1Mean3model/basemodel/stream_0_drop_2/Identity_1:output:0Jmodel/basemodel/global_average_pooling1d/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:         @21
/model/basemodel/global_average_pooling1d/Mean_1╨
*model/basemodel/dense_1_dropout/Identity_1Identity8model/basemodel/global_average_pooling1d/Mean_1:output:0*
T0*'
_output_shapes
:         @2,
*model/basemodel/dense_1_dropout/Identity_1┘
/model/basemodel/dense_1/MatMul_1/ReadVariableOpReadVariableOp6model_basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype021
/model/basemodel/dense_1/MatMul_1/ReadVariableOpю
 model/basemodel/dense_1/MatMul_1MatMul3model/basemodel/dense_1_dropout/Identity_1:output:07model/basemodel/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2"
 model/basemodel/dense_1/MatMul_1╪
0model/basemodel/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp7model_basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype022
0model/basemodel/dense_1/BiasAdd_1/ReadVariableOpщ
!model/basemodel/dense_1/BiasAdd_1BiasAdd*model/basemodel/dense_1/MatMul_1:product:08model/basemodel/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2#
!model/basemodel/dense_1/BiasAdd_1И
@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpReadVariableOpGmodel_basemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02B
@model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp╖
7model/basemodel/batch_normalization_2/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:29
7model/basemodel/batch_normalization_2/batchnorm_1/add/yи
5model/basemodel/batch_normalization_2/batchnorm_1/addAddV2Hmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp:value:0@model/basemodel/batch_normalization_2/batchnorm_1/add/y:output:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_2/batchnorm_1/add█
7model/basemodel/batch_normalization_2/batchnorm_1/RsqrtRsqrt9model/basemodel/batch_normalization_2/batchnorm_1/add:z:0*
T0*
_output_shapes
:T29
7model/basemodel/batch_normalization_2/batchnorm_1/RsqrtФ
Dmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpReadVariableOpKmodel_basemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02F
Dmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpе
5model/basemodel/batch_normalization_2/batchnorm_1/mulMul;model/basemodel/batch_normalization_2/batchnorm_1/Rsqrt:y:0Lmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_2/batchnorm_1/mulТ
7model/basemodel/batch_normalization_2/batchnorm_1/mul_1Mul*model/basemodel/dense_1/BiasAdd_1:output:09model/basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*'
_output_shapes
:         T29
7model/basemodel/batch_normalization_2/batchnorm_1/mul_1О
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02D
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1е
7model/basemodel/batch_normalization_2/batchnorm_1/mul_2MulJmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1:value:09model/basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*
_output_shapes
:T29
7model/basemodel/batch_normalization_2/batchnorm_1/mul_2О
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2ReadVariableOpImodel_basemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02D
Bmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2г
5model/basemodel/batch_normalization_2/batchnorm_1/subSubJmodel/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2:value:0;model/basemodel/batch_normalization_2/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:T27
5model/basemodel/batch_normalization_2/batchnorm_1/subе
7model/basemodel/batch_normalization_2/batchnorm_1/add_1AddV2;model/basemodel/batch_normalization_2/batchnorm_1/mul_1:z:09model/basemodel/batch_normalization_2/batchnorm_1/sub:z:0*
T0*'
_output_shapes
:         T29
7model/basemodel/batch_normalization_2/batchnorm_1/add_1╓
,model/basemodel/dense_activation_1/Sigmoid_1Sigmoid;model/basemodel/batch_normalization_2/batchnorm_1/add_1:z:0*
T0*'
_output_shapes
:         T2.
,model/basemodel/dense_activation_1/Sigmoid_1├
model/distance/subSub.model/basemodel/dense_activation_1/Sigmoid:y:00model/basemodel/dense_activation_1/Sigmoid_1:y:0*
T0*'
_output_shapes
:         T2
model/distance/subВ
model/distance/SquareSquaremodel/distance/sub:z:0*
T0*'
_output_shapes
:         T2
model/distance/SquareЧ
$model/distance/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2&
$model/distance/Sum/reduction_indices╝
model/distance/SumSummodel/distance/Square:y:0-model/distance/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2
model/distance/Sumq
model/distance/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/distance/Constй
model/distance/MaximumMaximummodel/distance/Sum:output:0model/distance/Const:output:0*
T0*'
_output_shapes
:         2
model/distance/MaximumА
model/distance/SqrtSqrtmodel/distance/Maximum:z:0*
T0*'
_output_shapes
:         2
model/distance/Sqrtr
IdentityIdentitymodel/distance/Sqrt:y:0^NoOp*
T0*'
_output_shapes
:         2

Identityф
NoOpNoOp=^model/basemodel/batch_normalization/batchnorm/ReadVariableOp?^model/basemodel/batch_normalization/batchnorm/ReadVariableOp_1?^model/basemodel/batch_normalization/batchnorm/ReadVariableOp_2A^model/basemodel/batch_normalization/batchnorm/mul/ReadVariableOp?^model/basemodel/batch_normalization/batchnorm_1/ReadVariableOpA^model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1A^model/basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2C^model/basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp?^model/basemodel/batch_normalization_1/batchnorm/ReadVariableOpA^model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1A^model/basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2C^model/basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpA^model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpC^model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1C^model/basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2E^model/basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp?^model/basemodel/batch_normalization_2/batchnorm/ReadVariableOpA^model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1A^model/basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2C^model/basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpA^model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpC^model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1C^model/basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2E^model/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp/^model/basemodel/dense_1/BiasAdd/ReadVariableOp1^model/basemodel/dense_1/BiasAdd_1/ReadVariableOp.^model/basemodel/dense_1/MatMul/ReadVariableOp0^model/basemodel/dense_1/MatMul_1/ReadVariableOp7^model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp9^model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpC^model/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpE^model/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp7^model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp9^model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpC^model/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpE^model/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:         ╓:         ╓: : : : : : : : : : : : : : : : : : 2|
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
Dmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpDmodel/basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp2`
.model/basemodel/dense_1/BiasAdd/ReadVariableOp.model/basemodel/dense_1/BiasAdd/ReadVariableOp2d
0model/basemodel/dense_1/BiasAdd_1/ReadVariableOp0model/basemodel/dense_1/BiasAdd_1/ReadVariableOp2^
-model/basemodel/dense_1/MatMul/ReadVariableOp-model/basemodel/dense_1/MatMul/ReadVariableOp2b
/model/basemodel/dense_1/MatMul_1/ReadVariableOp/model/basemodel/dense_1/MatMul_1/ReadVariableOp2p
6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp6model/basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2t
8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp8model/basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2И
Bmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpBmodel/basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2М
Dmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpDmodel/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2p
6model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp6model/basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp2t
8model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp8model/basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp2И
Bmodel/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpBmodel/basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2М
Dmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpDmodel/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp:Y U
,
_output_shapes
:         ╓
%
_user_specified_nameleft_inputs:ZV
,
_output_shapes
:         ╓
&
_user_specified_nameright_inputs
▌k
╝
F__inference_basemodel_layer_call_and_return_conditional_losses_5260463
inputs_0-
stream_0_conv_1_5260395: %
stream_0_conv_1_5260397: )
batch_normalization_5260400: )
batch_normalization_5260402: )
batch_normalization_5260404: )
batch_normalization_5260406: -
stream_0_conv_2_5260411: @%
stream_0_conv_2_5260413:@+
batch_normalization_1_5260416:@+
batch_normalization_1_5260418:@+
batch_normalization_1_5260420:@+
batch_normalization_1_5260422:@!
dense_1_5260429:@T
dense_1_5260431:T+
batch_normalization_2_5260434:T+
batch_normalization_2_5260436:T+
batch_normalization_2_5260438:T+
batch_normalization_2_5260440:T
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв'dense_1_dropout/StatefulPartitionedCallв'stream_0_conv_1/StatefulPartitionedCallв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв'stream_0_conv_2/StatefulPartitionedCallв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpв'stream_0_drop_1/StatefulPartitionedCallв'stream_0_drop_2/StatefulPartitionedCallв+stream_0_input_drop/StatefulPartitionedCallШ
+stream_0_input_drop/StatefulPartitionedCallStatefulPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_52601212-
+stream_0_input_drop/StatefulPartitionedCallЁ
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall4stream_0_input_drop/StatefulPartitionedCall:output:0stream_0_conv_1_5260395stream_0_conv_1_5260397*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_52596302)
'stream_0_conv_1/StatefulPartitionedCall╝
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_5260400batch_normalization_5260402batch_normalization_5260404batch_normalization_5260406*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_52600802-
+batch_normalization/StatefulPartitionedCallС
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_52596702
activation/PartitionedCall╒
'stream_0_drop_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0,^stream_0_input_drop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_52600222)
'stream_0_drop_1/StatefulPartitionedCallь
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall0stream_0_drop_1/StatefulPartitionedCall:output:0stream_0_conv_2_5260411stream_0_conv_2_5260413*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_52597002)
'stream_0_conv_2/StatefulPartitionedCall╩
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_5260416batch_normalization_1_5260418batch_normalization_1_5260420batch_normalization_1_5260422*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_52599812/
-batch_normalization_1/StatefulPartitionedCallЩ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_52597402
activation_1/PartitionedCall╙
'stream_0_drop_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0(^stream_0_drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_52599232)
'stream_0_drop_2/StatefulPartitionedCall▓
(global_average_pooling1d/PartitionedCallPartitionedCall0stream_0_drop_2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_52597542*
(global_average_pooling1d/PartitionedCall┌
'dense_1_dropout/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0(^stream_0_drop_2/StatefulPartitionedCall*
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
GPU2*0J 8В *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_52598952)
'dense_1_dropout/StatefulPartitionedCall┐
dense_1/StatefulPartitionedCallStatefulPartitionedCall0dense_1_dropout/StatefulPartitionedCall:output:0dense_1_5260429dense_1_5260431*
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
GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_52597792!
dense_1/StatefulPartitionedCall╜
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_5260434batch_normalization_2_5260436batch_normalization_2_5260438batch_normalization_2_5260440*
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
GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_52595172/
-batch_normalization_2/StatefulPartitionedCallж
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_52597992$
"dense_activation_1/PartitionedCall╩
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_5260395*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
&stream_0_conv_1/kernel/Regularizer/mul╨
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_5260411*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulо
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_5260429*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

Identityб
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^dense_1_dropout/StatefulPartitionedCall(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp(^stream_0_drop_1/StatefulPartitionedCall(^stream_0_drop_2/StatefulPartitionedCall,^stream_0_input_drop/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ╓: : : : : : : : : : : : : : : : : : 2Z
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
:         ╓
"
_user_specified_name
inputs_0
┐
k
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_5259895

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┴
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seed╖2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
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
╕
▒
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5259271

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
╕
▒
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5263115

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
т
╙
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_5262853

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
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
:         ╓2
conv1d/ExpandDims╕
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
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓ *
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ╓ *
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓ 2	
BiasAdd▐
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
:         ╓ 2

Identity─
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╓: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:T P
,
_output_shapes
:         ╓
 
_user_specified_nameinputs
Э
в
1__inference_stream_0_conv_1_layer_call_fn_5262862

inputs
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_52596302
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ╓ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╓: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ╓
 
_user_specified_nameinputs
╞
j
1__inference_stream_0_drop_2_layer_call_fn_5263292

inputs
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_52599232
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ╓@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ╓@
 
_user_specified_nameinputs
ё
╥
7__inference_batch_normalization_1_layer_call_fn_5263242

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_52597252
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ╓@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ╓@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ╓@
 
_user_specified_nameinputs
Н
п
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5262936

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓ 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ╓ 2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ╓ 2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ╓ : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ╓ 
 
_user_specified_nameinputs
Л	
╨
5__inference_batch_normalization_layer_call_fn_5262996

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_52591692
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                   2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
э
╨
5__inference_batch_normalization_layer_call_fn_5263009

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_52596552
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ╓ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ╓ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ╓ 
 
_user_specified_nameinputs
╞
V
*__inference_distance_layer_call_fn_5262799
inputs_0
inputs_1
identity╙
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_distance_layer_call_and_return_conditional_losses_52607292
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         T:         T:Q M
'
_output_shapes
:         T
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         T
"
_user_specified_name
inputs/1
Ў
▒
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5263392

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
б
╦
'__inference_model_layer_call_fn_5262013
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
identityИвStatefulPartitionedCall╙
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
:         *.
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_52611162
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:         ╓:         ╓: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:         ╓
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:         ╓
"
_user_specified_name
inputs/1
мё
 
B__inference_model_layer_call_and_return_conditional_losses_5261607
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
identityИв6basemodel/batch_normalization/batchnorm/ReadVariableOpв8basemodel/batch_normalization/batchnorm/ReadVariableOp_1в8basemodel/batch_normalization/batchnorm/ReadVariableOp_2в:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpв8basemodel/batch_normalization/batchnorm_1/ReadVariableOpв:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1в:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2в<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpв8basemodel/batch_normalization_1/batchnorm/ReadVariableOpв:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1в:basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2в<basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOpв:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpв<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1в<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2в>basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOpв8basemodel/batch_normalization_2/batchnorm/ReadVariableOpв:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1в:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2в<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpв:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpв<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1в<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2в>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpв(basemodel/dense_1/BiasAdd/ReadVariableOpв*basemodel/dense_1/BiasAdd_1/ReadVariableOpв'basemodel/dense_1/MatMul/ReadVariableOpв)basemodel/dense_1/MatMul_1/ReadVariableOpв0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpв2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpв<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpв>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpв0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpв2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpв<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpв>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpЭ
&basemodel/stream_0_input_drop/IdentityIdentityinputs_0*
T0*,
_output_shapes
:         ╓2(
&basemodel/stream_0_input_drop/Identityн
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
:         ╓2-
+basemodel/stream_0_conv_1/conv1d/ExpandDimsЖ
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2/
-basemodel/stream_0_conv_1/conv1d/ExpandDims_1Я
 basemodel/stream_0_conv_1/conv1dConv2D4basemodel/stream_0_conv_1/conv1d/ExpandDims:output:06basemodel/stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓ *
paddingSAME*
strides
2"
 basemodel/stream_0_conv_1/conv1dс
(basemodel/stream_0_conv_1/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ╓ *
squeeze_dims

¤        2*
(basemodel/stream_0_conv_1/conv1d/Squeeze┌
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOpї
!basemodel/stream_0_conv_1/BiasAddBiasAdd1basemodel/stream_0_conv_1/conv1d/Squeeze:output:08basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓ 2#
!basemodel/stream_0_conv_1/BiasAddь
6basemodel/batch_normalization/batchnorm/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
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
: 2-
+basemodel/batch_normalization/batchnorm/add╜
-basemodel/batch_normalization/batchnorm/RsqrtRsqrt/basemodel/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-basemodel/batch_normalization/batchnorm/Rsqrt°
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:basemodel/batch_normalization/batchnorm/mul/ReadVariableOp¤
+basemodel/batch_normalization/batchnorm/mulMul1basemodel/batch_normalization/batchnorm/Rsqrt:y:0Bbasemodel/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+basemodel/batch_normalization/batchnorm/mul∙
-basemodel/batch_normalization/batchnorm/mul_1Mul*basemodel/stream_0_conv_1/BiasAdd:output:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓ 2/
-basemodel/batch_normalization/batchnorm/mul_1Є
8basemodel/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8basemodel/batch_normalization/batchnorm/ReadVariableOp_1¤
-basemodel/batch_normalization/batchnorm/mul_2Mul@basemodel/batch_normalization/batchnorm/ReadVariableOp_1:value:0/basemodel/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-basemodel/batch_normalization/batchnorm/mul_2Є
8basemodel/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8basemodel/batch_normalization/batchnorm/ReadVariableOp_2√
+basemodel/batch_normalization/batchnorm/subSub@basemodel/batch_normalization/batchnorm/ReadVariableOp_2:value:01basemodel/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+basemodel/batch_normalization/batchnorm/subВ
-basemodel/batch_normalization/batchnorm/add_1AddV21basemodel/batch_normalization/batchnorm/mul_1:z:0/basemodel/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ╓ 2/
-basemodel/batch_normalization/batchnorm/add_1и
basemodel/activation/ReluRelu1basemodel/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ╓ 2
basemodel/activation/Relu┤
"basemodel/stream_0_drop_1/IdentityIdentity'basemodel/activation/Relu:activations:0*
T0*,
_output_shapes
:         ╓ 2$
"basemodel/stream_0_drop_1/Identityн
/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        21
/basemodel/stream_0_conv_2/conv1d/ExpandDims/dimК
+basemodel/stream_0_conv_2/conv1d/ExpandDims
ExpandDims+basemodel/stream_0_drop_1/Identity:output:08basemodel/stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ╓ 2-
+basemodel/stream_0_conv_2/conv1d/ExpandDimsЖ
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02>
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpи
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
: @2/
-basemodel/stream_0_conv_2/conv1d/ExpandDims_1Я
 basemodel/stream_0_conv_2/conv1dConv2D4basemodel/stream_0_conv_2/conv1d/ExpandDims:output:06basemodel/stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓@*
paddingSAME*
strides
2"
 basemodel/stream_0_conv_2/conv1dс
(basemodel/stream_0_conv_2/conv1d/SqueezeSqueeze)basemodel/stream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:         ╓@*
squeeze_dims

¤        2*
(basemodel/stream_0_conv_2/conv1d/Squeeze┌
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOpї
!basemodel/stream_0_conv_2/BiasAddBiasAdd1basemodel/stream_0_conv_2/conv1d/Squeeze:output:08basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓@2#
!basemodel/stream_0_conv_2/BiasAddЄ
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
/basemodel/batch_normalization_1/batchnorm/mul_1Mul*basemodel/stream_0_conv_2/BiasAdd:output:01basemodel/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓@21
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
:         ╓@21
/basemodel/batch_normalization_1/batchnorm/add_1о
basemodel/activation_1/ReluRelu3basemodel/batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ╓@2
basemodel/activation_1/Relu╢
"basemodel/stream_0_drop_2/IdentityIdentity)basemodel/activation_1/Relu:activations:0*
T0*,
_output_shapes
:         ╓@2$
"basemodel/stream_0_drop_2/Identity╕
9basemodel/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9basemodel/global_average_pooling1d/Mean/reduction_indices¤
'basemodel/global_average_pooling1d/MeanMean+basemodel/stream_0_drop_2/Identity:output:0Bbasemodel/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2)
'basemodel/global_average_pooling1d/Mean╕
"basemodel/dense_1_dropout/IdentityIdentity0basemodel/global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:         @2$
"basemodel/dense_1_dropout/Identity├
'basemodel/dense_1/MatMul/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
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
8basemodel/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
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
:T2/
-basemodel/batch_normalization_2/batchnorm/add├
/basemodel/batch_normalization_2/batchnorm/RsqrtRsqrt1basemodel/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_2/batchnorm/Rsqrt■
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOpЕ
-basemodel/batch_normalization_2/batchnorm/mulMul3basemodel/batch_normalization_2/batchnorm/Rsqrt:y:0Dbasemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_2/batchnorm/mulЄ
/basemodel/batch_normalization_2/batchnorm/mul_1Mul"basemodel/dense_1/BiasAdd:output:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T21
/basemodel/batch_normalization_2/batchnorm/mul_1°
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1Е
/basemodel/batch_normalization_2/batchnorm/mul_2MulBbasemodel/batch_normalization_2/batchnorm/ReadVariableOp_1:value:01basemodel/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_2/batchnorm/mul_2°
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2Г
-basemodel/batch_normalization_2/batchnorm/subSubBbasemodel/batch_normalization_2/batchnorm/ReadVariableOp_2:value:03basemodel/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2/
-basemodel/batch_normalization_2/batchnorm/subЕ
/basemodel/batch_normalization_2/batchnorm/add_1AddV23basemodel/batch_normalization_2/batchnorm/mul_1:z:01basemodel/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T21
/basemodel/batch_normalization_2/batchnorm/add_1╛
$basemodel/dense_activation_1/SigmoidSigmoid3basemodel/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2&
$basemodel/dense_activation_1/Sigmoidб
(basemodel/stream_0_input_drop/Identity_1Identityinputs_1*
T0*,
_output_shapes
:         ╓2*
(basemodel/stream_0_input_drop/Identity_1▒
1basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        23
1basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dimЦ
-basemodel/stream_0_conv_1/conv1d_1/ExpandDims
ExpandDims1basemodel/stream_0_input_drop/Identity_1:output:0:basemodel/stream_0_conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ╓2/
-basemodel/stream_0_conv_1/conv1d_1/ExpandDimsК
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02@
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOpм
3basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dimз
/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 21
/basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1з
"basemodel/stream_0_conv_1/conv1d_1Conv2D6basemodel/stream_0_conv_1/conv1d_1/ExpandDims:output:08basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓ *
paddingSAME*
strides
2$
"basemodel/stream_0_conv_1/conv1d_1ч
*basemodel/stream_0_conv_1/conv1d_1/SqueezeSqueeze+basemodel/stream_0_conv_1/conv1d_1:output:0*
T0*,
_output_shapes
:         ╓ *
squeeze_dims

¤        2,
*basemodel/stream_0_conv_1/conv1d_1/Squeeze▐
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp¤
#basemodel/stream_0_conv_1/BiasAdd_1BiasAdd3basemodel/stream_0_conv_1/conv1d_1/Squeeze:output:0:basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓ 2%
#basemodel/stream_0_conv_1/BiasAdd_1Ё
8basemodel/batch_normalization/batchnorm_1/ReadVariableOpReadVariableOp?basemodel_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02:
8basemodel/batch_normalization/batchnorm_1/ReadVariableOpз
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
: 2/
-basemodel/batch_normalization/batchnorm_1/add├
/basemodel/batch_normalization/batchnorm_1/RsqrtRsqrt1basemodel/batch_normalization/batchnorm_1/add:z:0*
T0*
_output_shapes
: 21
/basemodel/batch_normalization/batchnorm_1/Rsqrt№
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpReadVariableOpCbasemodel_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02>
<basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOpЕ
-basemodel/batch_normalization/batchnorm_1/mulMul3basemodel/batch_normalization/batchnorm_1/Rsqrt:y:0Dbasemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2/
-basemodel/batch_normalization/batchnorm_1/mulБ
/basemodel/batch_normalization/batchnorm_1/mul_1Mul,basemodel/stream_0_conv_1/BiasAdd_1:output:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:         ╓ 21
/basemodel/batch_normalization/batchnorm_1/mul_1Ў
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1Е
/basemodel/batch_normalization/batchnorm_1/mul_2MulBbasemodel/batch_normalization/batchnorm_1/ReadVariableOp_1:value:01basemodel/batch_normalization/batchnorm_1/mul:z:0*
T0*
_output_shapes
: 21
/basemodel/batch_normalization/batchnorm_1/mul_2Ў
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2ReadVariableOpAbasemodel_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02<
:basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2Г
-basemodel/batch_normalization/batchnorm_1/subSubBbasemodel/batch_normalization/batchnorm_1/ReadVariableOp_2:value:03basemodel/batch_normalization/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
: 2/
-basemodel/batch_normalization/batchnorm_1/subК
/basemodel/batch_normalization/batchnorm_1/add_1AddV23basemodel/batch_normalization/batchnorm_1/mul_1:z:01basemodel/batch_normalization/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:         ╓ 21
/basemodel/batch_normalization/batchnorm_1/add_1о
basemodel/activation/Relu_1Relu3basemodel/batch_normalization/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:         ╓ 2
basemodel/activation/Relu_1║
$basemodel/stream_0_drop_1/Identity_1Identity)basemodel/activation/Relu_1:activations:0*
T0*,
_output_shapes
:         ╓ 2&
$basemodel/stream_0_drop_1/Identity_1▒
1basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        23
1basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dimТ
-basemodel/stream_0_conv_2/conv1d_1/ExpandDims
ExpandDims-basemodel/stream_0_drop_1/Identity_1:output:0:basemodel/stream_0_conv_2/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ╓ 2/
-basemodel/stream_0_conv_2/conv1d_1/ExpandDimsК
>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02@
>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOpм
3basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dimз
/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1
ExpandDimsFbasemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp:value:0<basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @21
/basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1з
"basemodel/stream_0_conv_2/conv1d_1Conv2D6basemodel/stream_0_conv_2/conv1d_1/ExpandDims:output:08basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓@*
paddingSAME*
strides
2$
"basemodel/stream_0_conv_2/conv1d_1ч
*basemodel/stream_0_conv_2/conv1d_1/SqueezeSqueeze+basemodel/stream_0_conv_2/conv1d_1:output:0*
T0*,
_output_shapes
:         ╓@*
squeeze_dims

¤        2,
*basemodel/stream_0_conv_2/conv1d_1/Squeeze▐
2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOpReadVariableOp9basemodel_stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp¤
#basemodel/stream_0_conv_2/BiasAdd_1BiasAdd3basemodel/stream_0_conv_2/conv1d_1/Squeeze:output:0:basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓@2%
#basemodel/stream_0_conv_2/BiasAdd_1Ў
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02<
:basemodel/batch_normalization_1/batchnorm_1/ReadVariableOpл
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
/basemodel/batch_normalization_1/batchnorm_1/add╔
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
/basemodel/batch_normalization_1/batchnorm_1/mulЗ
1basemodel/batch_normalization_1/batchnorm_1/mul_1Mul,basemodel/stream_0_conv_2/BiasAdd_1:output:03basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*,
_output_shapes
:         ╓@23
1basemodel/batch_normalization_1/batchnorm_1/mul_1№
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1Н
1basemodel/batch_normalization_1/batchnorm_1/mul_2MulDbasemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1:value:03basemodel/batch_normalization_1/batchnorm_1/mul:z:0*
T0*
_output_shapes
:@23
1basemodel/batch_normalization_1/batchnorm_1/mul_2№
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02>
<basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2Л
/basemodel/batch_normalization_1/batchnorm_1/subSubDbasemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2:value:05basemodel/batch_normalization_1/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:@21
/basemodel/batch_normalization_1/batchnorm_1/subТ
1basemodel/batch_normalization_1/batchnorm_1/add_1AddV25basemodel/batch_normalization_1/batchnorm_1/mul_1:z:03basemodel/batch_normalization_1/batchnorm_1/sub:z:0*
T0*,
_output_shapes
:         ╓@23
1basemodel/batch_normalization_1/batchnorm_1/add_1┤
basemodel/activation_1/Relu_1Relu5basemodel/batch_normalization_1/batchnorm_1/add_1:z:0*
T0*,
_output_shapes
:         ╓@2
basemodel/activation_1/Relu_1╝
$basemodel/stream_0_drop_2/Identity_1Identity+basemodel/activation_1/Relu_1:activations:0*
T0*,
_output_shapes
:         ╓@2&
$basemodel/stream_0_drop_2/Identity_1╝
;basemodel/global_average_pooling1d/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;basemodel/global_average_pooling1d/Mean_1/reduction_indicesЕ
)basemodel/global_average_pooling1d/Mean_1Mean-basemodel/stream_0_drop_2/Identity_1:output:0Dbasemodel/global_average_pooling1d/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:         @2+
)basemodel/global_average_pooling1d/Mean_1╛
$basemodel/dense_1_dropout/Identity_1Identity2basemodel/global_average_pooling1d/Mean_1:output:0*
T0*'
_output_shapes
:         @2&
$basemodel/dense_1_dropout/Identity_1╟
)basemodel/dense_1/MatMul_1/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02+
)basemodel/dense_1/MatMul_1/ReadVariableOp╓
basemodel/dense_1/MatMul_1MatMul-basemodel/dense_1_dropout/Identity_1:output:01basemodel/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
basemodel/dense_1/MatMul_1╞
*basemodel/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp1basemodel_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02,
*basemodel/dense_1/BiasAdd_1/ReadVariableOp╤
basemodel/dense_1/BiasAdd_1BiasAdd$basemodel/dense_1/MatMul_1:product:02basemodel/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         T2
basemodel/dense_1/BiasAdd_1Ў
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpReadVariableOpAbasemodel_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype02<
:basemodel/batch_normalization_2/batchnorm_1/ReadVariableOpл
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
:T21
/basemodel/batch_normalization_2/batchnorm_1/add╔
1basemodel/batch_normalization_2/batchnorm_1/RsqrtRsqrt3basemodel/batch_normalization_2/batchnorm_1/add:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_2/batchnorm_1/RsqrtВ
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpReadVariableOpEbasemodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype02@
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOpН
/basemodel/batch_normalization_2/batchnorm_1/mulMul5basemodel/batch_normalization_2/batchnorm_1/Rsqrt:y:0Fbasemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_2/batchnorm_1/mul·
1basemodel/batch_normalization_2/batchnorm_1/mul_1Mul$basemodel/dense_1/BiasAdd_1:output:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*'
_output_shapes
:         T23
1basemodel/batch_normalization_2/batchnorm_1/mul_1№
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1Н
1basemodel/batch_normalization_2/batchnorm_1/mul_2MulDbasemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1:value:03basemodel/batch_normalization_2/batchnorm_1/mul:z:0*
T0*
_output_shapes
:T23
1basemodel/batch_normalization_2/batchnorm_1/mul_2№
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2ReadVariableOpCbasemodel_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype02>
<basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2Л
/basemodel/batch_normalization_2/batchnorm_1/subSubDbasemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2:value:05basemodel/batch_normalization_2/batchnorm_1/mul_2:z:0*
T0*
_output_shapes
:T21
/basemodel/batch_normalization_2/batchnorm_1/subН
1basemodel/batch_normalization_2/batchnorm_1/add_1AddV25basemodel/batch_normalization_2/batchnorm_1/mul_1:z:03basemodel/batch_normalization_2/batchnorm_1/sub:z:0*
T0*'
_output_shapes
:         T23
1basemodel/batch_normalization_2/batchnorm_1/add_1─
&basemodel/dense_activation_1/Sigmoid_1Sigmoid5basemodel/batch_normalization_2/batchnorm_1/add_1:z:0*
T0*'
_output_shapes
:         T2(
&basemodel/dense_activation_1/Sigmoid_1л
distance/subSub(basemodel/dense_activation_1/Sigmoid:y:0*basemodel/dense_activation_1/Sigmoid_1:y:0*
T0*'
_output_shapes
:         T2
distance/subp
distance/SquareSquaredistance/sub:z:0*
T0*'
_output_shapes
:         T2
distance/SquareЛ
distance/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2 
distance/Sum/reduction_indicesд
distance/SumSumdistance/Square:y:0'distance/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
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
:         2
distance/Maximumn
distance/SqrtSqrtdistance/Maximum:z:0*
T0*'
_output_shapes
:         2
distance/Sqrt°
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpEbasemodel_stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul╧
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp0basemodel_dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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
IdentityIdentitydistance/Sqrt:y:0^NoOp*
T0*'
_output_shapes
:         2

Identityп
NoOpNoOp7^basemodel/batch_normalization/batchnorm/ReadVariableOp9^basemodel/batch_normalization/batchnorm/ReadVariableOp_19^basemodel/batch_normalization/batchnorm/ReadVariableOp_2;^basemodel/batch_normalization/batchnorm/mul/ReadVariableOp9^basemodel/batch_normalization/batchnorm_1/ReadVariableOp;^basemodel/batch_normalization/batchnorm_1/ReadVariableOp_1;^basemodel/batch_normalization/batchnorm_1/ReadVariableOp_2=^basemodel/batch_normalization/batchnorm_1/mul/ReadVariableOp9^basemodel/batch_normalization_1/batchnorm/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_1/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_1/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp=^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_1=^basemodel/batch_normalization_1/batchnorm_1/ReadVariableOp_2?^basemodel/batch_normalization_1/batchnorm_1/mul/ReadVariableOp9^basemodel/batch_normalization_2/batchnorm/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_1;^basemodel/batch_normalization_2/batchnorm/ReadVariableOp_2=^basemodel/batch_normalization_2/batchnorm/mul/ReadVariableOp;^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp=^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_1=^basemodel/batch_normalization_2/batchnorm_1/ReadVariableOp_2?^basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp)^basemodel/dense_1/BiasAdd/ReadVariableOp+^basemodel/dense_1/BiasAdd_1/ReadVariableOp(^basemodel/dense_1/MatMul/ReadVariableOp*^basemodel/dense_1/MatMul_1/ReadVariableOp1^basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp3^basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp=^basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp1^basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp3^basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp=^basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp?^basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:         ╓:         ╓: : : : : : : : : : : : : : : : : : 2p
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
>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp>basemodel/batch_normalization_2/batchnorm_1/mul/ReadVariableOp2T
(basemodel/dense_1/BiasAdd/ReadVariableOp(basemodel/dense_1/BiasAdd/ReadVariableOp2X
*basemodel/dense_1/BiasAdd_1/ReadVariableOp*basemodel/dense_1/BiasAdd_1/ReadVariableOp2R
'basemodel/dense_1/MatMul/ReadVariableOp'basemodel/dense_1/MatMul/ReadVariableOp2V
)basemodel/dense_1/MatMul_1/ReadVariableOp)basemodel/dense_1/MatMul_1/ReadVariableOp2d
0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_1/BiasAdd/ReadVariableOp2h
2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2basemodel/stream_0_conv_1/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp2А
>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_0_conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2d
0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp0basemodel/stream_0_conv_2/BiasAdd/ReadVariableOp2h
2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp2basemodel/stream_0_conv_2/BiasAdd_1/ReadVariableOp2|
<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp<basemodel/stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp2А
>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp>basemodel/stream_0_conv_2/conv1d_1/ExpandDims_1/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:V R
,
_output_shapes
:         ╓
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:         ╓
"
_user_specified_name
inputs/1
·
o
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_5262816

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
:         ╓2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ╓*
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
:         ╓2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ╓2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ╓2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓:T P
,
_output_shapes
:         ╓
 
_user_specified_nameinputs
║	
o
E__inference_distance_layer_call_and_return_conditional_losses_5260647

inputs
inputs_1
identityU
subSubinputsinputs_1*
T0*'
_output_shapes
:         T2
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:         T2
Squarey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2
Sum/reduction_indicesА
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
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
:         2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:         2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         T:         T:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs:OK
'
_output_shapes
:         T
 
_user_specified_nameinputs
┬	
q
E__inference_distance_layer_call_and_return_conditional_losses_5262775
inputs_0
inputs_1
identityW
subSubinputs_0inputs_1*
T0*'
_output_shapes
:         T2
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:         T2
Squarey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2
Sum/reduction_indicesА
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
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
:         2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:         2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         T:         T:Q M
'
_output_shapes
:         T
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         T
"
_user_specified_name
inputs/1
щ
k
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_5259799

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
Ў
k
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_5259923

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ╓@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ╓@*
dtype0*
seed╖*
seed2╖2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ╓@2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ╓@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ╓@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓@:T P
,
_output_shapes
:         ╓@
 
_user_specified_nameinputs
▒d
Р

F__inference_basemodel_layer_call_and_return_conditional_losses_5260391
inputs_0-
stream_0_conv_1_5260323: %
stream_0_conv_1_5260325: )
batch_normalization_5260328: )
batch_normalization_5260330: )
batch_normalization_5260332: )
batch_normalization_5260334: -
stream_0_conv_2_5260339: @%
stream_0_conv_2_5260341:@+
batch_normalization_1_5260344:@+
batch_normalization_1_5260346:@+
batch_normalization_1_5260348:@+
batch_normalization_1_5260350:@!
dense_1_5260357:@T
dense_1_5260359:T+
batch_normalization_2_5260362:T+
batch_normalization_2_5260364:T+
batch_normalization_2_5260366:T+
batch_normalization_2_5260368:T
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв'stream_0_conv_1/StatefulPartitionedCallв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв'stream_0_conv_2/StatefulPartitionedCallв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpА
#stream_0_input_drop/PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_52596072%
#stream_0_input_drop/PartitionedCallш
'stream_0_conv_1/StatefulPartitionedCallStatefulPartitionedCall,stream_0_input_drop/PartitionedCall:output:0stream_0_conv_1_5260323stream_0_conv_1_5260325*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_52596302)
'stream_0_conv_1/StatefulPartitionedCall╛
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_1/StatefulPartitionedCall:output:0batch_normalization_5260328batch_normalization_5260330batch_normalization_5260332batch_normalization_5260334*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_52596552-
+batch_normalization/StatefulPartitionedCallС
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_52596702
activation/PartitionedCallП
stream_0_drop_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_52596772!
stream_0_drop_1/PartitionedCallф
'stream_0_conv_2/StatefulPartitionedCallStatefulPartitionedCall(stream_0_drop_1/PartitionedCall:output:0stream_0_conv_2_5260339stream_0_conv_2_5260341*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_52597002)
'stream_0_conv_2/StatefulPartitionedCall╠
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall0stream_0_conv_2/StatefulPartitionedCall:output:0batch_normalization_1_5260344batch_normalization_1_5260346batch_normalization_1_5260348batch_normalization_1_5260350*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_52597252/
-batch_normalization_1/StatefulPartitionedCallЩ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_52597402
activation_1/PartitionedCallС
stream_0_drop_2/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_52597472!
stream_0_drop_2/PartitionedCallк
(global_average_pooling1d/PartitionedCallPartitionedCall(stream_0_drop_2/PartitionedCall:output:0*
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
GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_52597542*
(global_average_pooling1d/PartitionedCallШ
dense_1_dropout/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
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
GPU2*0J 8В *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_52597612!
dense_1_dropout/PartitionedCall╖
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1_dropout/PartitionedCall:output:0dense_1_5260357dense_1_5260359*
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
GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_52597792!
dense_1/StatefulPartitionedCall┐
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_5260362batch_normalization_2_5260364batch_normalization_2_5260366batch_normalization_2_5260368*
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
GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_52594572/
-batch_normalization_2/StatefulPartitionedCallж
"dense_activation_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_52597992$
"dense_activation_1/PartitionedCall╩
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpstream_0_conv_1_5260323*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
&stream_0_conv_1/kernel/Regularizer/mul╨
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpstream_0_conv_2_5260339*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulо
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_1_5260357*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

Identityї
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall.^dense_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_1/StatefulPartitionedCall6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp(^stream_0_conv_2/StatefulPartitionedCall9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ╓: : : : : : : : : : : : : : : : : : 2Z
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
:         ╓
"
_user_specified_name
inputs_0
П	
╥
7__inference_batch_normalization_1_layer_call_fn_5263229

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallк
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
GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_52593312
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
а
┴
+__inference_basemodel_layer_call_fn_5262763
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
identityИвStatefulPartitionedCall╠
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
:         T*.
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_52609522
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
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ╓: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:         ╓
"
_user_specified_name
inputs/0
·
o
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_5260121

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
:         ╓2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╘
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ╓*
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
:         ╓2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ╓2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ╓2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓:T P
,
_output_shapes
:         ╓
 
_user_specified_nameinputs
ї
e
I__inference_activation_1_layer_call_and_return_conditional_losses_5259740

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         ╓@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         ╓@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓@:T P
,
_output_shapes
:         ╓@
 
_user_specified_nameinputs
ж
┴
+__inference_basemodel_layer_call_fn_5262722
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
identityИвStatefulPartitionedCall╥
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
:         T*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_52605782
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
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ╓: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:         ╓
"
_user_specified_name
inputs/0
▌
J
.__inference_activation_1_layer_call_fn_5263265

inputs
identity╧
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_52597402
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ╓@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓@:T P
,
_output_shapes
:         ╓@
 
_user_specified_nameinputs
Г
╓
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_5259700

inputsA
+conv1d_expanddims_1_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpy
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
:         ╓ 2
conv1d/ExpandDims╕
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
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1╖
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓@*
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ╓@*
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
:         ╓@2	
BiasAddф
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mulp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:         ╓@2

Identity╟
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╓ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:         ╓ 
 
_user_specified_nameinputs
З
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5263304

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
:         ╓@:T P
,
_output_shapes
:         ╓@
 
_user_specified_nameinputs
┌
╥
7__inference_batch_normalization_2_layer_call_fn_5263452

inputs
unknown:T
	unknown_0:T
	unknown_1:T
	unknown_2:T
identityИвStatefulPartitionedCallЭ
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
GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_52595172
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
Ф;
┬
B__inference_model_layer_call_and_return_conditional_losses_5261116

inputs
inputs_1'
basemodel_5261040: 
basemodel_5261042: 
basemodel_5261044: 
basemodel_5261046: 
basemodel_5261048: 
basemodel_5261050: '
basemodel_5261052: @
basemodel_5261054:@
basemodel_5261056:@
basemodel_5261058:@
basemodel_5261060:@
basemodel_5261062:@#
basemodel_5261064:@T
basemodel_5261066:T
basemodel_5261068:T
basemodel_5261070:T
basemodel_5261072:T
basemodel_5261074:T
identityИв!basemodel/StatefulPartitionedCallв#basemodel/StatefulPartitionedCall_1в-dense_1/kernel/Regularizer/Abs/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpщ
!basemodel/StatefulPartitionedCallStatefulPartitionedCallinputsbasemodel_5261040basemodel_5261042basemodel_5261044basemodel_5261046basemodel_5261048basemodel_5261050basemodel_5261052basemodel_5261054basemodel_5261056basemodel_5261058basemodel_5261060basemodel_5261062basemodel_5261064basemodel_5261066basemodel_5261068basemodel_5261070basemodel_5261072basemodel_5261074*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*.
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_52609522#
!basemodel/StatefulPartitionedCallУ
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1basemodel_5261040basemodel_5261042basemodel_5261044basemodel_5261046basemodel_5261048basemodel_5261050basemodel_5261052basemodel_5261054basemodel_5261056basemodel_5261058basemodel_5261060basemodel_5261062basemodel_5261064basemodel_5261066basemodel_5261068basemodel_5261070basemodel_5261072basemodel_5261074"^basemodel/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*.
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_52609522%
#basemodel/StatefulPartitionedCall_1л
distance/PartitionedCallPartitionedCall*basemodel/StatefulPartitionedCall:output:0,basemodel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_distance_layer_call_and_return_conditional_losses_52607292
distance/PartitionedCall─
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_5261040*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
&stream_0_conv_1/kernel/Regularizer/mul╩
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_5261052*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul░
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_5261064*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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
dense_1/kernel/Regularizer/mul|
IdentityIdentity!distance/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity╗
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:         ╓:         ╓: : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:         ╓
 
_user_specified_nameinputs:TP
,
_output_shapes
:         ╓
 
_user_specified_nameinputs
я
╥
7__inference_batch_normalization_1_layer_call_fn_5263255

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_52599812
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ╓@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ╓@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ╓@
 
_user_specified_nameinputs
н╒
о 
#__inference__traced_restore_5263823
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
identity_50ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9Ъ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*ж
valueЬBЩ2B+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЄ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesи
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*▐
_output_shapes╦
╚::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	2
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

Identity_3е
AssignVariableOp_3AssignVariableOp assignvariableop_3_learning_rateIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4б
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5о
AssignVariableOp_5AssignVariableOp)assignvariableop_5_stream_0_conv_1_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6м
AssignVariableOp_6AssignVariableOp'assignvariableop_6_stream_0_conv_1_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7▒
AssignVariableOp_7AssignVariableOp,assignvariableop_7_batch_normalization_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8░
AssignVariableOp_8AssignVariableOp+assignvariableop_8_batch_normalization_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9о
AssignVariableOp_9AssignVariableOp)assignvariableop_9_stream_0_conv_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10░
AssignVariableOp_10AssignVariableOp(assignvariableop_10_stream_0_conv_2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11╖
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_1_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12╢
AssignVariableOp_12AssignVariableOp.assignvariableop_12_batch_normalization_1_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13к
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_1_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14и
AssignVariableOp_14AssignVariableOp assignvariableop_14_dense_1_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15╖
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_2_gammaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16╢
AssignVariableOp_16AssignVariableOp.assignvariableop_16_batch_normalization_2_betaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17╗
AssignVariableOp_17AssignVariableOp3assignvariableop_17_batch_normalization_moving_meanIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18┐
AssignVariableOp_18AssignVariableOp7assignvariableop_18_batch_normalization_moving_varianceIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19╜
AssignVariableOp_19AssignVariableOp5assignvariableop_19_batch_normalization_1_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20┴
AssignVariableOp_20AssignVariableOp9assignvariableop_20_batch_normalization_1_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21╜
AssignVariableOp_21AssignVariableOp5assignvariableop_21_batch_normalization_2_moving_meanIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22┴
AssignVariableOp_22AssignVariableOp9assignvariableop_22_batch_normalization_2_moving_varianceIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23б
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24б
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25╣
AssignVariableOp_25AssignVariableOp1assignvariableop_25_adam_stream_0_conv_1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26╖
AssignVariableOp_26AssignVariableOp/assignvariableop_26_adam_stream_0_conv_1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╝
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_batch_normalization_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28╗
AssignVariableOp_28AssignVariableOp3assignvariableop_28_adam_batch_normalization_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29╣
AssignVariableOp_29AssignVariableOp1assignvariableop_29_adam_stream_0_conv_2_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30╖
AssignVariableOp_30AssignVariableOp/assignvariableop_30_adam_stream_0_conv_2_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31╛
AssignVariableOp_31AssignVariableOp6assignvariableop_31_adam_batch_normalization_1_gamma_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32╜
AssignVariableOp_32AssignVariableOp5assignvariableop_32_adam_batch_normalization_1_beta_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33▒
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34п
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35╛
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_batch_normalization_2_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36╜
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_batch_normalization_2_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37╣
AssignVariableOp_37AssignVariableOp1assignvariableop_37_adam_stream_0_conv_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38╖
AssignVariableOp_38AssignVariableOp/assignvariableop_38_adam_stream_0_conv_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39╝
AssignVariableOp_39AssignVariableOp4assignvariableop_39_adam_batch_normalization_gamma_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40╗
AssignVariableOp_40AssignVariableOp3assignvariableop_40_adam_batch_normalization_beta_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41╣
AssignVariableOp_41AssignVariableOp1assignvariableop_41_adam_stream_0_conv_2_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42╖
AssignVariableOp_42AssignVariableOp/assignvariableop_42_adam_stream_0_conv_2_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43╛
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_batch_normalization_1_gamma_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44╜
AssignVariableOp_44AssignVariableOp5assignvariableop_44_adam_batch_normalization_1_beta_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45▒
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_1_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46п
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_1_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47╛
AssignVariableOp_47AssignVariableOp6assignvariableop_47_adam_batch_normalization_2_gamma_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48╜
AssignVariableOp_48AssignVariableOp5assignvariableop_48_adam_batch_normalization_2_beta_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_489
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpФ	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_49f
Identity_50IdentityIdentity_49:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_50№
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
·
к
__inference_loss_fn_2_5263495H
6dense_1_kernel_regularizer_abs_readvariableop_resource:@T
identityИв-dense_1/kernel/Regularizer/Abs/ReadVariableOp╒
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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
Ў
▒
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5259457

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
Ж+
щ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5260080

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
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
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         ╓ 2
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
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
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
╫#<2
AssignMovingAvg/decayд
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
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
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
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
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ╓ 2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ╓ 2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ╓ : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ╓ 
 
_user_specified_nameinputs
╗
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5259419

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
▒
j
1__inference_dense_1_dropout_layer_call_fn_5263341

inputs
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
GPU2*0J 8В *U
fPRN
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_52598952
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
_construction_contextkEagerRuntime*&
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Н
j
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_5259747

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ╓@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ╓@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓@:T P
,
_output_shapes
:         ╓@
 
_user_specified_nameinputs
╞
j
1__inference_stream_0_drop_1_layer_call_fn_5263059

inputs
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_52600222
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ╓ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓ 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ╓ 
 
_user_specified_nameinputs
ю
е
D__inference_dense_1_layer_call_and_return_conditional_losses_5263363

inputs0
matmul_readvariableop_resource:@T-
biasadd_readvariableop_resource:T
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@T*
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
BiasAdd╜
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
С	
╥
7__inference_batch_normalization_1_layer_call_fn_5263216

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallм
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
GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_52592712
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
╦
╛
__inference_loss_fn_0_5263473T
>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource: 
identityИв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpё
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp>stream_0_conv_1_kernel_regularizer_abs_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
Н
j
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_5263270

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ╓@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ╓@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓@:T P
,
_output_shapes
:         ╓@
 
_user_specified_nameinputs
║	
o
E__inference_distance_layer_call_and_return_conditional_losses_5260729

inputs
inputs_1
identityU
subSubinputsinputs_1*
T0*'
_output_shapes
:         T2
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:         T2
Squarey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2
Sum/reduction_indicesА
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
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
:         2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:         2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         T:         T:O K
'
_output_shapes
:         T
 
_user_specified_nameinputs:OK
'
_output_shapes
:         T
 
_user_specified_nameinputs
Э
в
1__inference_stream_0_conv_2_layer_call_fn_5263095

inputs
unknown: @
	unknown_0:@
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_52597002
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ╓@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╓ : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ╓ 
 
_user_specified_nameinputs
И+
ы
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5263203

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
:         ╓@2
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
:         ╓@2
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
:         ╓@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ╓@2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ╓@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ╓@
 
_user_specified_nameinputs
Н
j
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_5259677

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ╓ 2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ╓ 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓ :T P
,
_output_shapes
:         ╓ 
 
_user_specified_nameinputs
▒;
╦
B__inference_model_layer_call_and_return_conditional_losses_5261357
left_inputs
right_inputs'
basemodel_5261281: 
basemodel_5261283: 
basemodel_5261285: 
basemodel_5261287: 
basemodel_5261289: 
basemodel_5261291: '
basemodel_5261293: @
basemodel_5261295:@
basemodel_5261297:@
basemodel_5261299:@
basemodel_5261301:@
basemodel_5261303:@#
basemodel_5261305:@T
basemodel_5261307:T
basemodel_5261309:T
basemodel_5261311:T
basemodel_5261313:T
basemodel_5261315:T
identityИв!basemodel/StatefulPartitionedCallв#basemodel/StatefulPartitionedCall_1в-dense_1/kernel/Regularizer/Abs/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpю
!basemodel/StatefulPartitionedCallStatefulPartitionedCallleft_inputsbasemodel_5261281basemodel_5261283basemodel_5261285basemodel_5261287basemodel_5261289basemodel_5261291basemodel_5261293basemodel_5261295basemodel_5261297basemodel_5261299basemodel_5261301basemodel_5261303basemodel_5261305basemodel_5261307basemodel_5261309basemodel_5261311basemodel_5261313basemodel_5261315*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*.
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_52609522#
!basemodel/StatefulPartitionedCallЧ
#basemodel/StatefulPartitionedCall_1StatefulPartitionedCallright_inputsbasemodel_5261281basemodel_5261283basemodel_5261285basemodel_5261287basemodel_5261289basemodel_5261291basemodel_5261293basemodel_5261295basemodel_5261297basemodel_5261299basemodel_5261301basemodel_5261303basemodel_5261305basemodel_5261307basemodel_5261309basemodel_5261311basemodel_5261313basemodel_5261315"^basemodel/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         T*.
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_basemodel_layer_call_and_return_conditional_losses_52609522%
#basemodel/StatefulPartitionedCall_1л
distance/PartitionedCallPartitionedCall*basemodel/StatefulPartitionedCall:output:0,basemodel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_distance_layer_call_and_return_conditional_losses_52607292
distance/PartitionedCall─
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_5261281*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
&stream_0_conv_1/kernel/Regularizer/mul╩
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpbasemodel_5261293*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul░
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpbasemodel_5261305*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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
dense_1/kernel/Regularizer/mul|
IdentityIdentity!distance/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity╗
NoOpNoOp"^basemodel/StatefulPartitionedCall$^basemodel/StatefulPartitionedCall_1.^dense_1/kernel/Regularizer/Abs/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:         ╓:         ╓: : : : : : : : : : : : : : : : : : 2F
!basemodel/StatefulPartitionedCall!basemodel/StatefulPartitionedCall2J
#basemodel/StatefulPartitionedCall_1#basemodel/StatefulPartitionedCall_12^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp2n
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp2t
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:Y U
,
_output_shapes
:         ╓
%
_user_specified_nameleft_inputs:ZV
,
_output_shapes
:         ╓
&
_user_specified_nameright_inputs
╣+
ы
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5259331

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
╖+
щ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5262916

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
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
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient▒
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                   2
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
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
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
╫#<2
AssignMovingAvg/decayд
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpШ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/subП
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
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
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpа
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/subЧ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
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
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                   2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                   2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :                   2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                   : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
С
n
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_5262804

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ╓2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ╓2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓:T P
,
_output_shapes
:         ╓
 
_user_specified_nameinputs
Ъ
╨
%__inference_signature_wrapper_5261425
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
identityИвStatefulPartitionedCall└
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
:         *4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__wrapped_model_52590852
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:         ╓:         ╓: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:         ╓
%
_user_specified_nameleft_inputs:ZV
,
_output_shapes
:         ╓
&
_user_specified_nameright_inputs
Є
─
__inference_loss_fn_1_5263484W
Astream_0_conv_2_kernel_regularizer_square_readvariableop_resource: @
identityИв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp·
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAstream_0_conv_2_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mult
IdentityIdentity*stream_0_conv_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityЙ
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
∙
j
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_5263319

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         @2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Дg
А
 __inference__traced_save_5263666
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
ShardedFilenameФ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*ж
valueЬBЩ2B+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesь
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices║
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop1savev2_stream_0_conv_1_kernel_read_readvariableop/savev2_stream_0_conv_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop1savev2_stream_0_conv_2_kernel_read_readvariableop/savev2_stream_0_conv_2_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_adam_stream_0_conv_1_kernel_m_read_readvariableop6savev2_adam_stream_0_conv_1_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop8savev2_adam_stream_0_conv_2_kernel_m_read_readvariableop6savev2_adam_stream_0_conv_2_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop8savev2_adam_stream_0_conv_1_kernel_v_read_readvariableop6savev2_adam_stream_0_conv_1_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop8savev2_adam_stream_0_conv_2_kernel_v_read_readvariableop6savev2_adam_stream_0_conv_2_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422	2
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

identity_1Identity_1:output:0*▀
_input_shapes═
╩: : : : : : : : : : : @:@:@:@:@T:T:T:T: : :@:@:T:T: : : : : : : @:@:@:@:@T:T:T:T: : : : : @:@:@:@:@T:T:T:T: 2(
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
ъ▒
┌
F__inference_basemodel_layer_call_and_return_conditional_losses_5262138

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
identityИв,batch_normalization/batchnorm/ReadVariableOpв.batch_normalization/batchnorm/ReadVariableOp_1в.batch_normalization/batchnorm/ReadVariableOp_2в0batch_normalization/batchnorm/mul/ReadVariableOpв.batch_normalization_1/batchnorm/ReadVariableOpв0batch_normalization_1/batchnorm/ReadVariableOp_1в0batch_normalization_1/batchnorm/ReadVariableOp_2в2batch_normalization_1/batchnorm/mul/ReadVariableOpв.batch_normalization_2/batchnorm/ReadVariableOpв0batch_normalization_2/batchnorm/ReadVariableOp_1в0batch_normalization_2/batchnorm/ReadVariableOp_2в2batch_normalization_2/batchnorm/mul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв&stream_0_conv_1/BiasAdd/ReadVariableOpв2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв&stream_0_conv_2/BiasAdd/ReadVariableOpв2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpЗ
stream_0_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:         ╓2
stream_0_input_drop/IdentityЩ
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
:         ╓2#
!stream_0_conv_1/conv1d/ExpandDimsш
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2%
#stream_0_conv_1/conv1d/ExpandDims_1ў
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓ *
paddingSAME*
strides
2
stream_0_conv_1/conv1d├
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ╓ *
squeeze_dims

¤        2 
stream_0_conv_1/conv1d/Squeeze╝
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp═
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓ 2
stream_0_conv_1/BiasAdd╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
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
: 2#
!batch_normalization/batchnorm/addЯ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp╒
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/mul╤
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓ 2%
#batch_normalization/batchnorm/mul_1╘
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1╒
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/mul_2╘
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2╙
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/sub┌
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ╓ 2%
#batch_normalization/batchnorm/add_1К
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ╓ 2
activation/ReluЦ
stream_0_drop_1/IdentityIdentityactivation/Relu:activations:0*
T0*,
_output_shapes
:         ╓ 2
stream_0_drop_1/IdentityЩ
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2'
%stream_0_conv_2/conv1d/ExpandDims/dimт
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/Identity:output:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ╓ 2#
!stream_0_conv_2/conv1d/ExpandDimsш
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dimў
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#stream_0_conv_2/conv1d/ExpandDims_1ў
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓@*
paddingSAME*
strides
2
stream_0_conv_2/conv1d├
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:         ╓@*
squeeze_dims

¤        2 
stream_0_conv_2/conv1d/Squeeze╝
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp═
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓@2
stream_0_conv_2/BiasAdd╘
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
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓@2'
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
:         ╓@2'
%batch_normalization_1/batchnorm/add_1Р
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ╓@2
activation_1/ReluШ
stream_0_drop_2/IdentityIdentityactivation_1/Relu:activations:0*
T0*,
_output_shapes
:         ╓@2
stream_0_drop_2/Identityд
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices╒
global_average_pooling1d/MeanMean!stream_0_drop_2/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2
global_average_pooling1d/MeanЪ
dense_1_dropout/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:         @2
dense_1_dropout/Identityе
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
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
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
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
:T2%
#batch_normalization_2/batchnorm/addе
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp▌
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/mul╩
%batch_normalization_2/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_2/batchnorm/mul_1┌
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1▌
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/mul_2┌
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2█
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/sub▌
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_2/batchnorm/add_1а
dense_activation_1/SigmoidSigmoid)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2
dense_activation_1/Sigmoidю
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul┼
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

Identity╩
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ╓: : : : : : : : : : : : : : : : : : 2\
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
:         ╓
 
_user_specified_nameinputs
Ъ╗
И
F__inference_basemodel_layer_call_and_return_conditional_losses_5260952

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
identityИв#batch_normalization/AssignMovingAvgв2batch_normalization/AssignMovingAvg/ReadVariableOpв%batch_normalization/AssignMovingAvg_1в4batch_normalization/AssignMovingAvg_1/ReadVariableOpв,batch_normalization/batchnorm/ReadVariableOpв0batch_normalization/batchnorm/mul/ReadVariableOpв%batch_normalization_1/AssignMovingAvgв4batch_normalization_1/AssignMovingAvg/ReadVariableOpв'batch_normalization_1/AssignMovingAvg_1в6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpв.batch_normalization_1/batchnorm/ReadVariableOpв2batch_normalization_1/batchnorm/mul/ReadVariableOpв%batch_normalization_2/AssignMovingAvgв4batch_normalization_2/AssignMovingAvg/ReadVariableOpв'batch_normalization_2/AssignMovingAvg_1в6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpв.batch_normalization_2/batchnorm/ReadVariableOpв2batch_normalization_2/batchnorm/mul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв&stream_0_conv_1/BiasAdd/ReadVariableOpв2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв&stream_0_conv_2/BiasAdd/ReadVariableOpв2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpЛ
!stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2#
!stream_0_input_drop/dropout/Const┤
stream_0_input_drop/dropout/MulMulinputs*stream_0_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:         ╓2!
stream_0_input_drop/dropout/Mul|
!stream_0_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/ShapeР
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:         ╓*
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
:         ╓2*
(stream_0_input_drop/dropout/GreaterEqual└
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓2"
 stream_0_input_drop/dropout/Cast╧
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:         ╓2#
!stream_0_input_drop/dropout/Mul_1Щ
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
:         ╓2#
!stream_0_conv_1/conv1d/ExpandDimsш
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2%
#stream_0_conv_1/conv1d/ExpandDims_1ў
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓ *
paddingSAME*
strides
2
stream_0_conv_1/conv1d├
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ╓ *
squeeze_dims

¤        2 
stream_0_conv_1/conv1d/Squeeze╝
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp═
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓ 2
stream_0_conv_1/BiasAdd╣
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indicesщ
 batch_normalization/moments/meanMean stream_0_conv_1/BiasAdd:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2"
 batch_normalization/moments/mean╝
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
: 2*
(batch_normalization/moments/StopGradient 
-batch_normalization/moments/SquaredDifferenceSquaredDifference stream_0_conv_1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:         ╓ 2/
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
: *
	keep_dims(2&
$batch_normalization/moments/variance╜
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2%
#batch_normalization/moments/Squeeze┼
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
: *
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
: *
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpш
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/sub▀
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2)
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
: *
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOpЁ
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/subч
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2+
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
: 2#
!batch_normalization/batchnorm/addЯ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp╒
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/mul╤
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓ 2%
#batch_normalization/batchnorm/mul_1╦
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/mul_2╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02.
,batch_normalization/batchnorm/ReadVariableOp╤
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/sub┌
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ╓ 2%
#batch_normalization/batchnorm/add_1К
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ╓ 2
activation/ReluГ
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
:         ╓ 2
stream_0_drop_1/dropout/MulЛ
stream_0_drop_1/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/ShapeД
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:         ╓ *
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
:         ╓ 2&
$stream_0_drop_1/dropout/GreaterEqual┤
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓ 2
stream_0_drop_1/dropout/Cast┐
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:         ╓ 2
stream_0_drop_1/dropout/Mul_1Щ
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2'
%stream_0_conv_2/conv1d/ExpandDims/dimт
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/dropout/Mul_1:z:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ╓ 2#
!stream_0_conv_2/conv1d/ExpandDimsш
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dimў
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#stream_0_conv_2/conv1d/ExpandDims_1ў
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓@*
paddingSAME*
strides
2
stream_0_conv_2/conv1d├
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:         ╓@*
squeeze_dims

¤        2 
stream_0_conv_2/conv1d/Squeeze╝
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp═
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓@2
stream_0_conv_2/BiasAdd╜
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indicesя
"batch_normalization_1/moments/meanMean stream_0_conv_2/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
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
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_0_conv_2/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:         ╓@21
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
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓@2'
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
:         ╓@2'
%batch_normalization_1/batchnorm/add_1Р
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ╓@2
activation_1/ReluГ
stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
stream_0_drop_2/dropout/Const┴
stream_0_drop_2/dropout/MulMulactivation_1/Relu:activations:0&stream_0_drop_2/dropout/Const:output:0*
T0*,
_output_shapes
:         ╓@2
stream_0_drop_2/dropout/MulН
stream_0_drop_2/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_2/dropout/ShapeД
4stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_2/dropout/Shape:output:0*
T0*,
_output_shapes
:         ╓@*
dtype0*
seed╖*
seed2╖26
4stream_0_drop_2/dropout/random_uniform/RandomUniformХ
&stream_0_drop_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2(
&stream_0_drop_2/dropout/GreaterEqual/yГ
$stream_0_drop_2/dropout/GreaterEqualGreaterEqual=stream_0_drop_2/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ╓@2&
$stream_0_drop_2/dropout/GreaterEqual┤
stream_0_drop_2/dropout/CastCast(stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓@2
stream_0_drop_2/dropout/Cast┐
stream_0_drop_2/dropout/Mul_1Mulstream_0_drop_2/dropout/Mul:z:0 stream_0_drop_2/dropout/Cast:y:0*
T0*,
_output_shapes
:         ╓@2
stream_0_drop_2/dropout/Mul_1д
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices╒
global_average_pooling1d/MeanMean!stream_0_drop_2/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2
global_average_pooling1d/MeanГ
dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dense_1_dropout/dropout/Const├
dense_1_dropout/dropout/MulMul&global_average_pooling1d/Mean:output:0&dense_1_dropout/dropout/Const:output:0*
T0*'
_output_shapes
:         @2
dense_1_dropout/dropout/MulФ
dense_1_dropout/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
:2
dense_1_dropout/dropout/Shapeё
4dense_1_dropout/dropout/random_uniform/RandomUniformRandomUniform&dense_1_dropout/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seed╖26
4dense_1_dropout/dropout/random_uniform/RandomUniformХ
&dense_1_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2(
&dense_1_dropout/dropout/GreaterEqual/y■
$dense_1_dropout/dropout/GreaterEqualGreaterEqual=dense_1_dropout/dropout/random_uniform/RandomUniform:output:0/dense_1_dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2&
$dense_1_dropout/dropout/GreaterEqualп
dense_1_dropout/dropout/CastCast(dense_1_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dense_1_dropout/dropout/Cast║
dense_1_dropout/dropout/Mul_1Muldense_1_dropout/dropout/Mul:z:0 dense_1_dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dense_1_dropout/dropout/Mul_1е
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02
dense_1/MatMul/ReadVariableOpж
dense_1/MatMulMatMul!dense_1_dropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
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
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_2/moments/mean/reduction_indicesу
"batch_normalization_2/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2$
"batch_normalization_2/moments/mean╛
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:T2,
*batch_normalization_2/moments/StopGradient°
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:         T21
/batch_normalization_2/moments/SquaredDifference╛
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_2/moments/variance/reduction_indicesК
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2(
&batch_normalization_2/moments/variance┬
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze╩
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2)
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
:T*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpЁ
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_2/AssignMovingAvg/subч
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2+
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
:T*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp°
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_2/AssignMovingAvg_1/subя
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2-
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
:T2%
#batch_normalization_2/batchnorm/addе
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp▌
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/mul╩
%batch_normalization_2/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_2/batchnorm/mul_1╙
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/mul_2╘
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp┘
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/sub▌
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_2/batchnorm/add_1а
dense_activation_1/SigmoidSigmoid)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2
dense_activation_1/Sigmoidю
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul┼
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

Identity┌

NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ╓: : : : : : : : : : : : : : : : : : 2J
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
:         ╓
 
_user_specified_nameinputs
П
▒
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5263169

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
:         ╓@2
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
:         ╓@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ╓@2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ╓@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ╓@
 
_user_specified_nameinputs
ы
V
:__inference_global_average_pooling1d_layer_call_fn_5263314

inputs
identity╓
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
GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_52597542
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓@:T P
,
_output_shapes
:         ╓@
 
_user_specified_nameinputs
Ё▒
▄
F__inference_basemodel_layer_call_and_return_conditional_losses_5262422
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
identityИв,batch_normalization/batchnorm/ReadVariableOpв.batch_normalization/batchnorm/ReadVariableOp_1в.batch_normalization/batchnorm/ReadVariableOp_2в0batch_normalization/batchnorm/mul/ReadVariableOpв.batch_normalization_1/batchnorm/ReadVariableOpв0batch_normalization_1/batchnorm/ReadVariableOp_1в0batch_normalization_1/batchnorm/ReadVariableOp_2в2batch_normalization_1/batchnorm/mul/ReadVariableOpв.batch_normalization_2/batchnorm/ReadVariableOpв0batch_normalization_2/batchnorm/ReadVariableOp_1в0batch_normalization_2/batchnorm/ReadVariableOp_2в2batch_normalization_2/batchnorm/mul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв&stream_0_conv_1/BiasAdd/ReadVariableOpв2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв&stream_0_conv_2/BiasAdd/ReadVariableOpв2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpЙ
stream_0_input_drop/IdentityIdentityinputs_0*
T0*,
_output_shapes
:         ╓2
stream_0_input_drop/IdentityЩ
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
:         ╓2#
!stream_0_conv_1/conv1d/ExpandDimsш
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2%
#stream_0_conv_1/conv1d/ExpandDims_1ў
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓ *
paddingSAME*
strides
2
stream_0_conv_1/conv1d├
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ╓ *
squeeze_dims

¤        2 
stream_0_conv_1/conv1d/Squeeze╝
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp═
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓ 2
stream_0_conv_1/BiasAdd╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
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
: 2#
!batch_normalization/batchnorm/addЯ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp╒
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/mul╤
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓ 2%
#batch_normalization/batchnorm/mul_1╘
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1╒
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/mul_2╘
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2╙
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/sub┌
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ╓ 2%
#batch_normalization/batchnorm/add_1К
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ╓ 2
activation/ReluЦ
stream_0_drop_1/IdentityIdentityactivation/Relu:activations:0*
T0*,
_output_shapes
:         ╓ 2
stream_0_drop_1/IdentityЩ
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2'
%stream_0_conv_2/conv1d/ExpandDims/dimт
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/Identity:output:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ╓ 2#
!stream_0_conv_2/conv1d/ExpandDimsш
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dimў
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#stream_0_conv_2/conv1d/ExpandDims_1ў
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓@*
paddingSAME*
strides
2
stream_0_conv_2/conv1d├
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:         ╓@*
squeeze_dims

¤        2 
stream_0_conv_2/conv1d/Squeeze╝
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp═
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓@2
stream_0_conv_2/BiasAdd╘
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
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓@2'
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
:         ╓@2'
%batch_normalization_1/batchnorm/add_1Р
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ╓@2
activation_1/ReluШ
stream_0_drop_2/IdentityIdentityactivation_1/Relu:activations:0*
T0*,
_output_shapes
:         ╓@2
stream_0_drop_2/Identityд
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices╒
global_average_pooling1d/MeanMean!stream_0_drop_2/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2
global_average_pooling1d/MeanЪ
dense_1_dropout/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:         @2
dense_1_dropout/Identityе
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
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
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
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
:T2%
#batch_normalization_2/batchnorm/addе
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp▌
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/mul╩
%batch_normalization_2/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_2/batchnorm/mul_1┌
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1▌
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/mul_2┌
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2█
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/sub▌
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_2/batchnorm/add_1а
dense_activation_1/SigmoidSigmoid)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2
dense_activation_1/Sigmoidю
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul┼
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

Identity╩
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ╓: : : : : : : : : : : : : : : : : : 2\
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
:         ╓
"
_user_specified_name
inputs/0
Н
j
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_5263037

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ╓ 2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ╓ 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓ :T P
,
_output_shapes
:         ╓ 
 
_user_specified_nameinputs
И+
ы
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5259981

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
:         ╓@2
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
:         ╓@2
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
:         ╓@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ╓@2

IdentityЄ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ╓@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ╓@
 
_user_specified_nameinputs
Ъ╗
И
F__inference_basemodel_layer_call_and_return_conditional_losses_5262315

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
identityИв#batch_normalization/AssignMovingAvgв2batch_normalization/AssignMovingAvg/ReadVariableOpв%batch_normalization/AssignMovingAvg_1в4batch_normalization/AssignMovingAvg_1/ReadVariableOpв,batch_normalization/batchnorm/ReadVariableOpв0batch_normalization/batchnorm/mul/ReadVariableOpв%batch_normalization_1/AssignMovingAvgв4batch_normalization_1/AssignMovingAvg/ReadVariableOpв'batch_normalization_1/AssignMovingAvg_1в6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpв.batch_normalization_1/batchnorm/ReadVariableOpв2batch_normalization_1/batchnorm/mul/ReadVariableOpв%batch_normalization_2/AssignMovingAvgв4batch_normalization_2/AssignMovingAvg/ReadVariableOpв'batch_normalization_2/AssignMovingAvg_1в6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpв.batch_normalization_2/batchnorm/ReadVariableOpв2batch_normalization_2/batchnorm/mul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв&stream_0_conv_1/BiasAdd/ReadVariableOpв2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв&stream_0_conv_2/BiasAdd/ReadVariableOpв2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpЛ
!stream_0_input_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2#
!stream_0_input_drop/dropout/Const┤
stream_0_input_drop/dropout/MulMulinputs*stream_0_input_drop/dropout/Const:output:0*
T0*,
_output_shapes
:         ╓2!
stream_0_input_drop/dropout/Mul|
!stream_0_input_drop/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2#
!stream_0_input_drop/dropout/ShapeР
8stream_0_input_drop/dropout/random_uniform/RandomUniformRandomUniform*stream_0_input_drop/dropout/Shape:output:0*
T0*,
_output_shapes
:         ╓*
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
:         ╓2*
(stream_0_input_drop/dropout/GreaterEqual└
 stream_0_input_drop/dropout/CastCast,stream_0_input_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓2"
 stream_0_input_drop/dropout/Cast╧
!stream_0_input_drop/dropout/Mul_1Mul#stream_0_input_drop/dropout/Mul:z:0$stream_0_input_drop/dropout/Cast:y:0*
T0*,
_output_shapes
:         ╓2#
!stream_0_input_drop/dropout/Mul_1Щ
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
:         ╓2#
!stream_0_conv_1/conv1d/ExpandDimsш
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2%
#stream_0_conv_1/conv1d/ExpandDims_1ў
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓ *
paddingSAME*
strides
2
stream_0_conv_1/conv1d├
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ╓ *
squeeze_dims

¤        2 
stream_0_conv_1/conv1d/Squeeze╝
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp═
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓ 2
stream_0_conv_1/BiasAdd╣
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indicesщ
 batch_normalization/moments/meanMean stream_0_conv_1/BiasAdd:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2"
 batch_normalization/moments/mean╝
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
: 2*
(batch_normalization/moments/StopGradient 
-batch_normalization/moments/SquaredDifferenceSquaredDifference stream_0_conv_1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:         ╓ 2/
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
: *
	keep_dims(2&
$batch_normalization/moments/variance╜
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2%
#batch_normalization/moments/Squeeze┼
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
: *
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
: *
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpш
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/sub▀
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2)
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
: *
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOpЁ
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/subч
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2+
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
: 2#
!batch_normalization/batchnorm/addЯ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp╒
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/mul╤
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓ 2%
#batch_normalization/batchnorm/mul_1╦
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/mul_2╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02.
,batch_normalization/batchnorm/ReadVariableOp╤
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/sub┌
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ╓ 2%
#batch_normalization/batchnorm/add_1К
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ╓ 2
activation/ReluГ
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
:         ╓ 2
stream_0_drop_1/dropout/MulЛ
stream_0_drop_1/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_1/dropout/ShapeД
4stream_0_drop_1/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_1/dropout/Shape:output:0*
T0*,
_output_shapes
:         ╓ *
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
:         ╓ 2&
$stream_0_drop_1/dropout/GreaterEqual┤
stream_0_drop_1/dropout/CastCast(stream_0_drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓ 2
stream_0_drop_1/dropout/Cast┐
stream_0_drop_1/dropout/Mul_1Mulstream_0_drop_1/dropout/Mul:z:0 stream_0_drop_1/dropout/Cast:y:0*
T0*,
_output_shapes
:         ╓ 2
stream_0_drop_1/dropout/Mul_1Щ
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2'
%stream_0_conv_2/conv1d/ExpandDims/dimт
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/dropout/Mul_1:z:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ╓ 2#
!stream_0_conv_2/conv1d/ExpandDimsш
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dimў
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#stream_0_conv_2/conv1d/ExpandDims_1ў
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓@*
paddingSAME*
strides
2
stream_0_conv_2/conv1d├
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:         ╓@*
squeeze_dims

¤        2 
stream_0_conv_2/conv1d/Squeeze╝
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp═
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓@2
stream_0_conv_2/BiasAdd╜
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indicesя
"batch_normalization_1/moments/meanMean stream_0_conv_2/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
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
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference stream_0_conv_2/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:         ╓@21
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
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓@2'
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
:         ╓@2'
%batch_normalization_1/batchnorm/add_1Р
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ╓@2
activation_1/ReluГ
stream_0_drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
stream_0_drop_2/dropout/Const┴
stream_0_drop_2/dropout/MulMulactivation_1/Relu:activations:0&stream_0_drop_2/dropout/Const:output:0*
T0*,
_output_shapes
:         ╓@2
stream_0_drop_2/dropout/MulН
stream_0_drop_2/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
stream_0_drop_2/dropout/ShapeД
4stream_0_drop_2/dropout/random_uniform/RandomUniformRandomUniform&stream_0_drop_2/dropout/Shape:output:0*
T0*,
_output_shapes
:         ╓@*
dtype0*
seed╖*
seed2╖26
4stream_0_drop_2/dropout/random_uniform/RandomUniformХ
&stream_0_drop_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2(
&stream_0_drop_2/dropout/GreaterEqual/yГ
$stream_0_drop_2/dropout/GreaterEqualGreaterEqual=stream_0_drop_2/dropout/random_uniform/RandomUniform:output:0/stream_0_drop_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ╓@2&
$stream_0_drop_2/dropout/GreaterEqual┤
stream_0_drop_2/dropout/CastCast(stream_0_drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╓@2
stream_0_drop_2/dropout/Cast┐
stream_0_drop_2/dropout/Mul_1Mulstream_0_drop_2/dropout/Mul:z:0 stream_0_drop_2/dropout/Cast:y:0*
T0*,
_output_shapes
:         ╓@2
stream_0_drop_2/dropout/Mul_1д
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices╒
global_average_pooling1d/MeanMean!stream_0_drop_2/dropout/Mul_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2
global_average_pooling1d/MeanГ
dense_1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dense_1_dropout/dropout/Const├
dense_1_dropout/dropout/MulMul&global_average_pooling1d/Mean:output:0&dense_1_dropout/dropout/Const:output:0*
T0*'
_output_shapes
:         @2
dense_1_dropout/dropout/MulФ
dense_1_dropout/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
:2
dense_1_dropout/dropout/Shapeё
4dense_1_dropout/dropout/random_uniform/RandomUniformRandomUniform&dense_1_dropout/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seed╖26
4dense_1_dropout/dropout/random_uniform/RandomUniformХ
&dense_1_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2(
&dense_1_dropout/dropout/GreaterEqual/y■
$dense_1_dropout/dropout/GreaterEqualGreaterEqual=dense_1_dropout/dropout/random_uniform/RandomUniform:output:0/dense_1_dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2&
$dense_1_dropout/dropout/GreaterEqualп
dense_1_dropout/dropout/CastCast(dense_1_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dense_1_dropout/dropout/Cast║
dense_1_dropout/dropout/Mul_1Muldense_1_dropout/dropout/Mul:z:0 dense_1_dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dense_1_dropout/dropout/Mul_1е
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02
dense_1/MatMul/ReadVariableOpж
dense_1/MatMulMatMul!dense_1_dropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
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
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_2/moments/mean/reduction_indicesу
"batch_normalization_2/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2$
"batch_normalization_2/moments/mean╛
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:T2,
*batch_normalization_2/moments/StopGradient°
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:         T21
/batch_normalization_2/moments/SquaredDifference╛
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_2/moments/variance/reduction_indicesК
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2(
&batch_normalization_2/moments/variance┬
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze╩
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2)
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
:T*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpЁ
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:T2+
)batch_normalization_2/AssignMovingAvg/subч
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:T2+
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
:T*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp°
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:T2-
+batch_normalization_2/AssignMovingAvg_1/subя
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:T2-
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
:T2%
#batch_normalization_2/batchnorm/addе
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp▌
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/mul╩
%batch_normalization_2/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_2/batchnorm/mul_1╙
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/mul_2╘
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp┘
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/sub▌
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_2/batchnorm/add_1а
dense_activation_1/SigmoidSigmoid)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2
dense_activation_1/Sigmoidю
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul┼
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

Identity┌

NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ╓: : : : : : : : : : : : : : : : : : 2J
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
:         ╓
 
_user_specified_nameinputs
П
▒
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5259725

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
:         ╓@2
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
:         ╓@2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ╓@2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ╓@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ╓@
 
_user_specified_nameinputs
ъ▒
┌
F__inference_basemodel_layer_call_and_return_conditional_losses_5260578

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
identityИв,batch_normalization/batchnorm/ReadVariableOpв.batch_normalization/batchnorm/ReadVariableOp_1в.batch_normalization/batchnorm/ReadVariableOp_2в0batch_normalization/batchnorm/mul/ReadVariableOpв.batch_normalization_1/batchnorm/ReadVariableOpв0batch_normalization_1/batchnorm/ReadVariableOp_1в0batch_normalization_1/batchnorm/ReadVariableOp_2в2batch_normalization_1/batchnorm/mul/ReadVariableOpв.batch_normalization_2/batchnorm/ReadVariableOpв0batch_normalization_2/batchnorm/ReadVariableOp_1в0batch_normalization_2/batchnorm/ReadVariableOp_2в2batch_normalization_2/batchnorm/mul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpв&stream_0_conv_1/BiasAdd/ReadVariableOpв2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpв5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpв&stream_0_conv_2/BiasAdd/ReadVariableOpв2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpв8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpЗ
stream_0_input_drop/IdentityIdentityinputs*
T0*,
_output_shapes
:         ╓2
stream_0_input_drop/IdentityЩ
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
:         ╓2#
!stream_0_conv_1/conv1d/ExpandDimsш
2stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2%
#stream_0_conv_1/conv1d/ExpandDims_1ў
stream_0_conv_1/conv1dConv2D*stream_0_conv_1/conv1d/ExpandDims:output:0,stream_0_conv_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓ *
paddingSAME*
strides
2
stream_0_conv_1/conv1d├
stream_0_conv_1/conv1d/SqueezeSqueezestream_0_conv_1/conv1d:output:0*
T0*,
_output_shapes
:         ╓ *
squeeze_dims

¤        2 
stream_0_conv_1/conv1d/Squeeze╝
&stream_0_conv_1/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&stream_0_conv_1/BiasAdd/ReadVariableOp═
stream_0_conv_1/BiasAddBiasAdd'stream_0_conv_1/conv1d/Squeeze:output:0.stream_0_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓ 2
stream_0_conv_1/BiasAdd╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
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
: 2#
!batch_normalization/batchnorm/addЯ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp╒
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/mul╤
#batch_normalization/batchnorm/mul_1Mul stream_0_conv_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓ 2%
#batch_normalization/batchnorm/mul_1╘
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1╒
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/mul_2╘
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2╙
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/sub┌
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ╓ 2%
#batch_normalization/batchnorm/add_1К
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ╓ 2
activation/ReluЦ
stream_0_drop_1/IdentityIdentityactivation/Relu:activations:0*
T0*,
_output_shapes
:         ╓ 2
stream_0_drop_1/IdentityЩ
%stream_0_conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2'
%stream_0_conv_2/conv1d/ExpandDims/dimт
!stream_0_conv_2/conv1d/ExpandDims
ExpandDims!stream_0_drop_1/Identity:output:0.stream_0_conv_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ╓ 2#
!stream_0_conv_2/conv1d/ExpandDimsш
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOpФ
'stream_0_conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stream_0_conv_2/conv1d/ExpandDims_1/dimў
#stream_0_conv_2/conv1d/ExpandDims_1
ExpandDims:stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:00stream_0_conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#stream_0_conv_2/conv1d/ExpandDims_1ў
stream_0_conv_2/conv1dConv2D*stream_0_conv_2/conv1d/ExpandDims:output:0,stream_0_conv_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ╓@*
paddingSAME*
strides
2
stream_0_conv_2/conv1d├
stream_0_conv_2/conv1d/SqueezeSqueezestream_0_conv_2/conv1d:output:0*
T0*,
_output_shapes
:         ╓@*
squeeze_dims

¤        2 
stream_0_conv_2/conv1d/Squeeze╝
&stream_0_conv_2/BiasAdd/ReadVariableOpReadVariableOp/stream_0_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&stream_0_conv_2/BiasAdd/ReadVariableOp═
stream_0_conv_2/BiasAddBiasAdd'stream_0_conv_2/conv1d/Squeeze:output:0.stream_0_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ╓@2
stream_0_conv_2/BiasAdd╘
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
%batch_normalization_1/batchnorm/mul_1Mul stream_0_conv_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓@2'
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
:         ╓@2'
%batch_normalization_1/batchnorm/add_1Р
activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:         ╓@2
activation_1/ReluШ
stream_0_drop_2/IdentityIdentityactivation_1/Relu:activations:0*
T0*,
_output_shapes
:         ╓@2
stream_0_drop_2/Identityд
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices╒
global_average_pooling1d/MeanMean!stream_0_drop_2/Identity:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @2
global_average_pooling1d/MeanЪ
dense_1_dropout/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:         @2
dense_1_dropout/Identityе
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
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
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:T*
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
:T2%
#batch_normalization_2/batchnorm/addе
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:T*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp▌
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/mul╩
%batch_normalization_2/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_2/batchnorm/mul_1┌
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:T*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1▌
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/mul_2┌
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:T*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2█
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/sub▌
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         T2'
%batch_normalization_2/batchnorm/add_1а
dense_activation_1/SigmoidSigmoid)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:         T2
dense_activation_1/Sigmoidю
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;stream_0_conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp├
&stream_0_conv_1/kernel/Regularizer/AbsAbs=stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2(
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
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;stream_0_conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp╧
)stream_0_conv_2/kernel/Regularizer/SquareSquare@stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2+
)stream_0_conv_2/kernel/Regularizer/Squareй
(stream_0_conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(stream_0_conv_2/kernel/Regularizer/Const┌
&stream_0_conv_2/kernel/Regularizer/SumSum-stream_0_conv_2/kernel/Regularizer/Square:y:01stream_0_conv_2/kernel/Regularizer/Const:output:0*
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
╫#<2*
(stream_0_conv_2/kernel/Regularizer/mul/x▄
&stream_0_conv_2/kernel/Regularizer/mulMul1stream_0_conv_2/kernel/Regularizer/mul/x:output:0/stream_0_conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&stream_0_conv_2/kernel/Regularizer/mul┼
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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

Identity╩
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^dense_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_1/BiasAdd/ReadVariableOp3^stream_0_conv_1/conv1d/ExpandDims_1/ReadVariableOp6^stream_0_conv_1/kernel/Regularizer/Abs/ReadVariableOp'^stream_0_conv_2/BiasAdd/ReadVariableOp3^stream_0_conv_2/conv1d/ExpandDims_1/ReadVariableOp9^stream_0_conv_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ╓: : : : : : : : : : : : : : : : : : 2\
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
:         ╓
 
_user_specified_nameinputs
┬	
q
E__inference_distance_layer_call_and_return_conditional_losses_5262787
inputs_0
inputs_1
identityW
subSubinputs_0inputs_1*
T0*'
_output_shapes
:         T2
subU
SquareSquaresub:z:0*
T0*'
_output_shapes
:         T2
Squarey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2
Sum/reduction_indicesА
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         *
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
:         2	
MaximumS
SqrtSqrtMaximum:z:0*
T0*'
_output_shapes
:         2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         T:         T:Q M
'
_output_shapes
:         T
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         T
"
_user_specified_name
inputs/1
З
q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5259754

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
:         ╓@:T P
,
_output_shapes
:         ╓@
 
_user_specified_nameinputs
ю
е
D__inference_dense_1_layer_call_and_return_conditional_losses_5259779

inputs0
matmul_readvariableop_resource:@T-
biasadd_readvariableop_resource:T
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв-dense_1/kernel/Regularizer/Abs/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@T*
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
BiasAdd╜
-dense_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@T*
dtype02/
-dense_1/kernel/Regularizer/Abs/ReadVariableOpз
dense_1/kernel/Regularizer/AbsAbs5dense_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@T2 
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
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_1/kernel/Regularizer/Abs/ReadVariableOp-dense_1/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ы
Q
5__inference_stream_0_input_drop_layer_call_fn_5262821

inputs
identity╓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_52596072
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ╓2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓:T P
,
_output_shapes
:         ╓
 
_user_specified_nameinputs
╣+
ы
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5263149

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
С
n
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_5259607

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ╓2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ╓2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓:T P
,
_output_shapes
:         ╓
 
_user_specified_nameinputs
ы
╨
5__inference_batch_normalization_layer_call_fn_5263022

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_52600802
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ╓ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ╓ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ╓ 
 
_user_specified_nameinputs
═*
ы
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5263426

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
┘
H
,__inference_activation_layer_call_fn_5263032

inputs
identity═
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╓ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_52596702
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ╓ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓ :T P
,
_output_shapes
:         ╓ 
 
_user_specified_nameinputs
Я
V
:__inference_global_average_pooling1d_layer_call_fn_5263309

inputs
identity▀
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
GPU2*0J 8В *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_52594192
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
╒
P
4__inference_dense_activation_1_layer_call_fn_5263462

inputs
identity╨
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
GPU2*0J 8В *X
fSRQ
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_52597992
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
є
c
G__inference_activation_layer_call_and_return_conditional_losses_5263027

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:         ╓ 2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:         ╓ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╓ :T P
,
_output_shapes
:         ╓ 
 
_user_specified_nameinputs
╢
╥
'__inference_model_layer_call_fn_5261197
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
identityИвStatefulPartitionedCall┌
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
:         *.
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_52611162
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:         ╓:         ╓: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:         ╓
%
_user_specified_nameleft_inputs:ZV
,
_output_shapes
:         ╓
&
_user_specified_nameright_inputs
Н
п
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5259655

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ╓ 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ╓ 2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:         ╓ 2

Identity┬
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ╓ : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         ╓ 
 
_user_specified_nameinputs
щ
k
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_5263457

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
 
_user_specified_nameinputs"иL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Д
serving_defaultЁ
H
left_inputs9
serving_default_left_inputs:0         ╓
J
right_inputs:
serving_default_right_inputs:0         ╓<
distance0
StatefulPartitionedCall:0         tensorflow/serving/predict:є╒
┤
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
ц_default_save_signature
+ч&call_and_return_all_conditional_losses
ш__call__"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Н
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
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer-14
regularization_losses
trainable_variables
	variables
	keras_api
+щ&call_and_return_all_conditional_losses
ъ__call__"
_tf_keras_network
з
regularization_losses
trainable_variables
 	variables
!	keras_api
+ы&call_and_return_all_conditional_losses
ь__call__"
_tf_keras_layer
├

"beta_1

#beta_2
	$decay
%learning_rate
&iter'm╬(m╧)m╨*m╤+m╥,m╙-m╘.m╒/m╓0m╫1m╪2m┘'v┌(v█)v▄*v▌+v▐,v▀-vр.vс/vт0vу1vф2vх"
	optimizer
 "
trackable_list_wrapper
v
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211"
trackable_list_wrapper
ж
'0
(1
)2
*3
34
45
+6
,7
-8
.9
510
611
/12
013
114
215
716
817"
trackable_list_wrapper
╬
9metrics

:layers
;layer_metrics
<layer_regularization_losses
regularization_losses
trainable_variables
=non_trainable_variables
	variables
ш__call__
ц_default_save_signature
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
-
эserving_default"
signature_map
"
_tf_keras_input_layer
з
>regularization_losses
?trainable_variables
@	variables
A	keras_api
+ю&call_and_return_all_conditional_losses
я__call__"
_tf_keras_layer
╜

'kernel
(bias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
+Ё&call_and_return_all_conditional_losses
ё__call__"
_tf_keras_layer
ь
Faxis
	)gamma
*beta
3moving_mean
4moving_variance
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
+Є&call_and_return_all_conditional_losses
є__call__"
_tf_keras_layer
з
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
+Ї&call_and_return_all_conditional_losses
ї__call__"
_tf_keras_layer
з
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
+Ў&call_and_return_all_conditional_losses
ў__call__"
_tf_keras_layer
╜

+kernel
,bias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
+°&call_and_return_all_conditional_losses
∙__call__"
_tf_keras_layer
ь
Waxis
	-gamma
.beta
5moving_mean
6moving_variance
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
+·&call_and_return_all_conditional_losses
√__call__"
_tf_keras_layer
з
\regularization_losses
]trainable_variables
^	variables
_	keras_api
+№&call_and_return_all_conditional_losses
¤__call__"
_tf_keras_layer
з
`regularization_losses
atrainable_variables
b	variables
c	keras_api
+■&call_and_return_all_conditional_losses
 __call__"
_tf_keras_layer
з
dregularization_losses
etrainable_variables
f	variables
g	keras_api
+А&call_and_return_all_conditional_losses
Б__call__"
_tf_keras_layer
з
hregularization_losses
itrainable_variables
j	variables
k	keras_api
+В&call_and_return_all_conditional_losses
Г__call__"
_tf_keras_layer
╜

/kernel
0bias
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
+Д&call_and_return_all_conditional_losses
Е__call__"
_tf_keras_layer
ь
paxis
	1gamma
2beta
7moving_mean
8moving_variance
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
+Ж&call_and_return_all_conditional_losses
З__call__"
_tf_keras_layer
з
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
+И&call_and_return_all_conditional_losses
Й__call__"
_tf_keras_layer
8
К0
Л1
М2"
trackable_list_wrapper
v
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211"
trackable_list_wrapper
ж
'0
(1
)2
*3
34
45
+6
,7
-8
.9
510
611
/12
013
114
215
716
817"
trackable_list_wrapper
░
ymetrics

zlayers
{layer_metrics
|layer_regularization_losses
regularization_losses
trainable_variables
}non_trainable_variables
	variables
ъ__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
│
~metrics

layers
Аlayer_metrics
 Бlayer_regularization_losses
regularization_losses
trainable_variables
Вnon_trainable_variables
 	variables
ь__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
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
(
Г0"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
30
41
52
63
74
85"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Дmetrics
Еlayers
Жlayer_metrics
 Зlayer_regularization_losses
>regularization_losses
?trainable_variables
Иnon_trainable_variables
@	variables
я__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
(
К0"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
╡
Йmetrics
Кlayers
Лlayer_metrics
 Мlayer_regularization_losses
Bregularization_losses
Ctrainable_variables
Нnon_trainable_variables
D	variables
ё__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
<
)0
*1
32
43"
trackable_list_wrapper
╡
Оmetrics
Пlayers
Рlayer_metrics
 Сlayer_regularization_losses
Gregularization_losses
Htrainable_variables
Тnon_trainable_variables
I	variables
є__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Уmetrics
Фlayers
Хlayer_metrics
 Цlayer_regularization_losses
Kregularization_losses
Ltrainable_variables
Чnon_trainable_variables
M	variables
ї__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Шmetrics
Щlayers
Ъlayer_metrics
 Ыlayer_regularization_losses
Oregularization_losses
Ptrainable_variables
Ьnon_trainable_variables
Q	variables
ў__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
(
Л0"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
╡
Эmetrics
Юlayers
Яlayer_metrics
 аlayer_regularization_losses
Sregularization_losses
Ttrainable_variables
бnon_trainable_variables
U	variables
∙__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
<
-0
.1
52
63"
trackable_list_wrapper
╡
вmetrics
гlayers
дlayer_metrics
 еlayer_regularization_losses
Xregularization_losses
Ytrainable_variables
жnon_trainable_variables
Z	variables
√__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
зmetrics
иlayers
йlayer_metrics
 кlayer_regularization_losses
\regularization_losses
]trainable_variables
лnon_trainable_variables
^	variables
¤__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
мmetrics
нlayers
оlayer_metrics
 пlayer_regularization_losses
`regularization_losses
atrainable_variables
░non_trainable_variables
b	variables
 __call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
▒metrics
▓layers
│layer_metrics
 ┤layer_regularization_losses
dregularization_losses
etrainable_variables
╡non_trainable_variables
f	variables
Б__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╢metrics
╖layers
╕layer_metrics
 ╣layer_regularization_losses
hregularization_losses
itrainable_variables
║non_trainable_variables
j	variables
Г__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
(
М0"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
╡
╗metrics
╝layers
╜layer_metrics
 ╛layer_regularization_losses
lregularization_losses
mtrainable_variables
┐non_trainable_variables
n	variables
Е__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
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
72
83"
trackable_list_wrapper
╡
└metrics
┴layers
┬layer_metrics
 ├layer_regularization_losses
qregularization_losses
rtrainable_variables
─non_trainable_variables
s	variables
З__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
┼metrics
╞layers
╟layer_metrics
 ╚layer_regularization_losses
uregularization_losses
vtrainable_variables
╔non_trainable_variables
w	variables
Й__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
О
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
14"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
30
41
52
63
74
85"
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
R

╩total

╦count
╠	variables
═	keras_api"
_tf_keras_metric
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
(
К0"
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
.
30
41"
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
(
Л0"
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
.
50
61"
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
(
М0"
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
.
70
81"
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
:  (2total
:  (2count
0
╩0
╦1"
trackable_list_wrapper
.
╠	variables"
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
▀B▄
"__inference__wrapped_model_5259085left_inputsright_inputs"Ш
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
╓2╙
B__inference_model_layer_call_and_return_conditional_losses_5261607
B__inference_model_layer_call_and_return_conditional_losses_5261929
B__inference_model_layer_call_and_return_conditional_losses_5261277
B__inference_model_layer_call_and_return_conditional_losses_5261357└
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
ъ2ч
'__inference_model_layer_call_fn_5260707
'__inference_model_layer_call_fn_5261971
'__inference_model_layer_call_fn_5262013
'__inference_model_layer_call_fn_5261197└
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
Ў2є
F__inference_basemodel_layer_call_and_return_conditional_losses_5262138
F__inference_basemodel_layer_call_and_return_conditional_losses_5262315
F__inference_basemodel_layer_call_and_return_conditional_losses_5260391
F__inference_basemodel_layer_call_and_return_conditional_losses_5260463
F__inference_basemodel_layer_call_and_return_conditional_losses_5262422
F__inference_basemodel_layer_call_and_return_conditional_losses_5262599└
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
╘2╤
+__inference_basemodel_layer_call_fn_5259859
+__inference_basemodel_layer_call_fn_5262640
+__inference_basemodel_layer_call_fn_5262681
+__inference_basemodel_layer_call_fn_5260319
+__inference_basemodel_layer_call_fn_5262722
+__inference_basemodel_layer_call_fn_5262763└
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
╘2╤
E__inference_distance_layer_call_and_return_conditional_losses_5262775
E__inference_distance_layer_call_and_return_conditional_losses_5262787└
╖▓│
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
kwonlydefaultsк 
annotationsк *
 
Ю2Ы
*__inference_distance_layer_call_fn_5262793
*__inference_distance_layer_call_fn_5262799└
╖▓│
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
kwonlydefaultsк 
annotationsк *
 
▄B┘
%__inference_signature_wrapper_5261425left_inputsright_inputs"Ф
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
▐2█
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_5262804
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_5262816┤
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
и2е
5__inference_stream_0_input_drop_layer_call_fn_5262821
5__inference_stream_0_input_drop_layer_call_fn_5262826┤
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
Ў2є
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_5262853в
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
█2╪
1__inference_stream_0_conv_1_layer_call_fn_5262862в
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
В2 
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5262882
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5262916
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5262936
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5262970┤
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
Ц2У
5__inference_batch_normalization_layer_call_fn_5262983
5__inference_batch_normalization_layer_call_fn_5262996
5__inference_batch_normalization_layer_call_fn_5263009
5__inference_batch_normalization_layer_call_fn_5263022┤
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
ё2ю
G__inference_activation_layer_call_and_return_conditional_losses_5263027в
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
╓2╙
,__inference_activation_layer_call_fn_5263032в
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
╓2╙
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_5263037
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_5263049┤
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
а2Э
1__inference_stream_0_drop_1_layer_call_fn_5263054
1__inference_stream_0_drop_1_layer_call_fn_5263059┤
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
Ў2є
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_5263086в
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
█2╪
1__inference_stream_0_conv_2_layer_call_fn_5263095в
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
К2З
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5263115
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5263149
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5263169
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5263203┤
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
Ю2Ы
7__inference_batch_normalization_1_layer_call_fn_5263216
7__inference_batch_normalization_1_layer_call_fn_5263229
7__inference_batch_normalization_1_layer_call_fn_5263242
7__inference_batch_normalization_1_layer_call_fn_5263255┤
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
є2Ё
I__inference_activation_1_layer_call_and_return_conditional_losses_5263260в
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
╪2╒
.__inference_activation_1_layer_call_fn_5263265в
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
╓2╙
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_5263270
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_5263282┤
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
а2Э
1__inference_stream_0_drop_2_layer_call_fn_5263287
1__inference_stream_0_drop_2_layer_call_fn_5263292┤
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
у2р
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5263298
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5263304п
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
н2к
:__inference_global_average_pooling1d_layer_call_fn_5263309
:__inference_global_average_pooling1d_layer_call_fn_5263314п
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
╓2╙
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_5263319
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_5263331┤
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
а2Э
1__inference_dense_1_dropout_layer_call_fn_5263336
1__inference_dense_1_dropout_layer_call_fn_5263341┤
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
ю2ы
D__inference_dense_1_layer_call_and_return_conditional_losses_5263363в
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
╙2╨
)__inference_dense_1_layer_call_fn_5263372в
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
т2▀
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5263392
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5263426┤
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
м2й
7__inference_batch_normalization_2_layer_call_fn_5263439
7__inference_batch_normalization_2_layer_call_fn_5263452┤
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
∙2Ў
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_5263457в
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
▐2█
4__inference_dense_activation_1_layer_call_fn_5263462в
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
┤2▒
__inference_loss_fn_0_5263473П
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
┤2▒
__inference_loss_fn_1_5263484П
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
┤2▒
__inference_loss_fn_2_5263495П
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
annotationsк *в ▌
"__inference__wrapped_model_5259085╢'(4)3*+,6-5./08172kвh
aв^
\ЪY
*К'
left_inputs         ╓
+К(
right_inputs         ╓
к "3к0
.
distance"К
distance         п
I__inference_activation_1_layer_call_and_return_conditional_losses_5263260b4в1
*в'
%К"
inputs         ╓@
к "*в'
 К
0         ╓@
Ъ З
.__inference_activation_1_layer_call_fn_5263265U4в1
*в'
%К"
inputs         ╓@
к "К         ╓@н
G__inference_activation_layer_call_and_return_conditional_losses_5263027b4в1
*в'
%К"
inputs         ╓ 
к "*в'
 К
0         ╓ 
Ъ Е
,__inference_activation_layer_call_fn_5263032U4в1
*в'
%К"
inputs         ╓ 
к "К         ╓ ┼
F__inference_basemodel_layer_call_and_return_conditional_losses_5260391{'(4)3*+,6-5./08172>в;
4в1
'К$
inputs_0         ╓
p 

 
к "%в"
К
0         T
Ъ ┼
F__inference_basemodel_layer_call_and_return_conditional_losses_5260463{'(34)*+,56-./07812>в;
4в1
'К$
inputs_0         ╓
p

 
к "%в"
К
0         T
Ъ ├
F__inference_basemodel_layer_call_and_return_conditional_losses_5262138y'(4)3*+,6-5./08172<в9
2в/
%К"
inputs         ╓
p 

 
к "%в"
К
0         T
Ъ ├
F__inference_basemodel_layer_call_and_return_conditional_losses_5262315y'(34)*+,56-./07812<в9
2в/
%К"
inputs         ╓
p

 
к "%в"
К
0         T
Ъ ╦
F__inference_basemodel_layer_call_and_return_conditional_losses_5262422А'(4)3*+,6-5./08172Cв@
9в6
,Ъ)
'К$
inputs/0         ╓
p 

 
к "%в"
К
0         T
Ъ ╦
F__inference_basemodel_layer_call_and_return_conditional_losses_5262599А'(34)*+,56-./07812Cв@
9в6
,Ъ)
'К$
inputs/0         ╓
p

 
к "%в"
К
0         T
Ъ Э
+__inference_basemodel_layer_call_fn_5259859n'(4)3*+,6-5./08172>в;
4в1
'К$
inputs_0         ╓
p 

 
к "К         TЭ
+__inference_basemodel_layer_call_fn_5260319n'(34)*+,56-./07812>в;
4в1
'К$
inputs_0         ╓
p

 
к "К         TЫ
+__inference_basemodel_layer_call_fn_5262640l'(4)3*+,6-5./08172<в9
2в/
%К"
inputs         ╓
p 

 
к "К         TЫ
+__inference_basemodel_layer_call_fn_5262681l'(34)*+,56-./07812<в9
2в/
%К"
inputs         ╓
p

 
к "К         Tв
+__inference_basemodel_layer_call_fn_5262722s'(4)3*+,6-5./08172Cв@
9в6
,Ъ)
'К$
inputs/0         ╓
p 

 
к "К         Tв
+__inference_basemodel_layer_call_fn_5262763s'(34)*+,56-./07812Cв@
9в6
,Ъ)
'К$
inputs/0         ╓
p

 
к "К         T╥
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5263115|6-5.@в=
6в3
-К*
inputs                  @
p 
к "2в/
(К%
0                  @
Ъ ╥
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5263149|56-.@в=
6в3
-К*
inputs                  @
p
к "2в/
(К%
0                  @
Ъ ┬
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5263169l6-5.8в5
.в+
%К"
inputs         ╓@
p 
к "*в'
 К
0         ╓@
Ъ ┬
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5263203l56-.8в5
.в+
%К"
inputs         ╓@
p
к "*в'
 К
0         ╓@
Ъ к
7__inference_batch_normalization_1_layer_call_fn_5263216o6-5.@в=
6в3
-К*
inputs                  @
p 
к "%К"                  @к
7__inference_batch_normalization_1_layer_call_fn_5263229o56-.@в=
6в3
-К*
inputs                  @
p
к "%К"                  @Ъ
7__inference_batch_normalization_1_layer_call_fn_5263242_6-5.8в5
.в+
%К"
inputs         ╓@
p 
к "К         ╓@Ъ
7__inference_batch_normalization_1_layer_call_fn_5263255_56-.8в5
.в+
%К"
inputs         ╓@
p
к "К         ╓@╕
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5263392b81723в0
)в&
 К
inputs         T
p 
к "%в"
К
0         T
Ъ ╕
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5263426b78123в0
)в&
 К
inputs         T
p
к "%в"
К
0         T
Ъ Р
7__inference_batch_normalization_2_layer_call_fn_5263439U81723в0
)в&
 К
inputs         T
p 
к "К         TР
7__inference_batch_normalization_2_layer_call_fn_5263452U78123в0
)в&
 К
inputs         T
p
к "К         T╨
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5262882|4)3*@в=
6в3
-К*
inputs                   
p 
к "2в/
(К%
0                   
Ъ ╨
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5262916|34)*@в=
6в3
-К*
inputs                   
p
к "2в/
(К%
0                   
Ъ └
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5262936l4)3*8в5
.в+
%К"
inputs         ╓ 
p 
к "*в'
 К
0         ╓ 
Ъ └
P__inference_batch_normalization_layer_call_and_return_conditional_losses_5262970l34)*8в5
.в+
%К"
inputs         ╓ 
p
к "*в'
 К
0         ╓ 
Ъ и
5__inference_batch_normalization_layer_call_fn_5262983o4)3*@в=
6в3
-К*
inputs                   
p 
к "%К"                   и
5__inference_batch_normalization_layer_call_fn_5262996o34)*@в=
6в3
-К*
inputs                   
p
к "%К"                   Ш
5__inference_batch_normalization_layer_call_fn_5263009_4)3*8в5
.в+
%К"
inputs         ╓ 
p 
к "К         ╓ Ш
5__inference_batch_normalization_layer_call_fn_5263022_34)*8в5
.в+
%К"
inputs         ╓ 
p
к "К         ╓ м
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_5263319\3в0
)в&
 К
inputs         @
p 
к "%в"
К
0         @
Ъ м
L__inference_dense_1_dropout_layer_call_and_return_conditional_losses_5263331\3в0
)в&
 К
inputs         @
p
к "%в"
К
0         @
Ъ Д
1__inference_dense_1_dropout_layer_call_fn_5263336O3в0
)в&
 К
inputs         @
p 
к "К         @Д
1__inference_dense_1_dropout_layer_call_fn_5263341O3в0
)в&
 К
inputs         @
p
к "К         @д
D__inference_dense_1_layer_call_and_return_conditional_losses_5263363\/0/в,
%в"
 К
inputs         @
к "%в"
К
0         T
Ъ |
)__inference_dense_1_layer_call_fn_5263372O/0/в,
%в"
 К
inputs         @
к "К         Tл
O__inference_dense_activation_1_layer_call_and_return_conditional_losses_5263457X/в,
%в"
 К
inputs         T
к "%в"
К
0         T
Ъ Г
4__inference_dense_activation_1_layer_call_fn_5263462K/в,
%в"
 К
inputs         T
к "К         T╒
E__inference_distance_layer_call_and_return_conditional_losses_5262775Лbв_
XвU
KЪH
"К
inputs/0         T
"К
inputs/1         T

 
p 
к "%в"
К
0         
Ъ ╒
E__inference_distance_layer_call_and_return_conditional_losses_5262787Лbв_
XвU
KЪH
"К
inputs/0         T
"К
inputs/1         T

 
p
к "%в"
К
0         
Ъ м
*__inference_distance_layer_call_fn_5262793~bв_
XвU
KЪH
"К
inputs/0         T
"К
inputs/1         T

 
p 
к "К         м
*__inference_distance_layer_call_fn_5262799~bв_
XвU
KЪH
"К
inputs/0         T
"К
inputs/1         T

 
p
к "К         ╘
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5263298{IвF
?в<
6К3
inputs'                           

 
к ".в+
$К!
0                  
Ъ ║
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5263304a8в5
.в+
%К"
inputs         ╓@

 
к "%в"
К
0         @
Ъ м
:__inference_global_average_pooling1d_layer_call_fn_5263309nIвF
?в<
6К3
inputs'                           

 
к "!К                  Т
:__inference_global_average_pooling1d_layer_call_fn_5263314T8в5
.в+
%К"
inputs         ╓@

 
к "К         @<
__inference_loss_fn_0_5263473'в

в 
к "К <
__inference_loss_fn_1_5263484+в

в 
к "К <
__inference_loss_fn_2_5263495/в

в 
к "К ў
B__inference_model_layer_call_and_return_conditional_losses_5261277░'(4)3*+,6-5./08172sвp
iвf
\ЪY
*К'
left_inputs         ╓
+К(
right_inputs         ╓
p 

 
к "%в"
К
0         
Ъ ў
B__inference_model_layer_call_and_return_conditional_losses_5261357░'(34)*+,56-./07812sвp
iвf
\ЪY
*К'
left_inputs         ╓
+К(
right_inputs         ╓
p

 
к "%в"
К
0         
Ъ Ё
B__inference_model_layer_call_and_return_conditional_losses_5261607й'(4)3*+,6-5./08172lвi
bв_
UЪR
'К$
inputs/0         ╓
'К$
inputs/1         ╓
p 

 
к "%в"
К
0         
Ъ Ё
B__inference_model_layer_call_and_return_conditional_losses_5261929й'(34)*+,56-./07812lвi
bв_
UЪR
'К$
inputs/0         ╓
'К$
inputs/1         ╓
p

 
к "%в"
К
0         
Ъ ╧
'__inference_model_layer_call_fn_5260707г'(4)3*+,6-5./08172sвp
iвf
\ЪY
*К'
left_inputs         ╓
+К(
right_inputs         ╓
p 

 
к "К         ╧
'__inference_model_layer_call_fn_5261197г'(34)*+,56-./07812sвp
iвf
\ЪY
*К'
left_inputs         ╓
+К(
right_inputs         ╓
p

 
к "К         ╚
'__inference_model_layer_call_fn_5261971Ь'(4)3*+,6-5./08172lвi
bв_
UЪR
'К$
inputs/0         ╓
'К$
inputs/1         ╓
p 

 
к "К         ╚
'__inference_model_layer_call_fn_5262013Ь'(34)*+,56-./07812lвi
bв_
UЪR
'К$
inputs/0         ╓
'К$
inputs/1         ╓
p

 
к "К         №
%__inference_signature_wrapper_5261425╥'(4)3*+,6-5./08172ЖвВ
в 
{кx
9
left_inputs*К'
left_inputs         ╓
;
right_inputs+К(
right_inputs         ╓"3к0
.
distance"К
distance         ╢
L__inference_stream_0_conv_1_layer_call_and_return_conditional_losses_5262853f'(4в1
*в'
%К"
inputs         ╓
к "*в'
 К
0         ╓ 
Ъ О
1__inference_stream_0_conv_1_layer_call_fn_5262862Y'(4в1
*в'
%К"
inputs         ╓
к "К         ╓ ╢
L__inference_stream_0_conv_2_layer_call_and_return_conditional_losses_5263086f+,4в1
*в'
%К"
inputs         ╓ 
к "*в'
 К
0         ╓@
Ъ О
1__inference_stream_0_conv_2_layer_call_fn_5263095Y+,4в1
*в'
%К"
inputs         ╓ 
к "К         ╓@╢
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_5263037f8в5
.в+
%К"
inputs         ╓ 
p 
к "*в'
 К
0         ╓ 
Ъ ╢
L__inference_stream_0_drop_1_layer_call_and_return_conditional_losses_5263049f8в5
.в+
%К"
inputs         ╓ 
p
к "*в'
 К
0         ╓ 
Ъ О
1__inference_stream_0_drop_1_layer_call_fn_5263054Y8в5
.в+
%К"
inputs         ╓ 
p 
к "К         ╓ О
1__inference_stream_0_drop_1_layer_call_fn_5263059Y8в5
.в+
%К"
inputs         ╓ 
p
к "К         ╓ ╢
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_5263270f8в5
.в+
%К"
inputs         ╓@
p 
к "*в'
 К
0         ╓@
Ъ ╢
L__inference_stream_0_drop_2_layer_call_and_return_conditional_losses_5263282f8в5
.в+
%К"
inputs         ╓@
p
к "*в'
 К
0         ╓@
Ъ О
1__inference_stream_0_drop_2_layer_call_fn_5263287Y8в5
.в+
%К"
inputs         ╓@
p 
к "К         ╓@О
1__inference_stream_0_drop_2_layer_call_fn_5263292Y8в5
.в+
%К"
inputs         ╓@
p
к "К         ╓@║
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_5262804f8в5
.в+
%К"
inputs         ╓
p 
к "*в'
 К
0         ╓
Ъ ║
P__inference_stream_0_input_drop_layer_call_and_return_conditional_losses_5262816f8в5
.в+
%К"
inputs         ╓
p
к "*в'
 К
0         ╓
Ъ Т
5__inference_stream_0_input_drop_layer_call_fn_5262821Y8в5
.в+
%К"
inputs         ╓
p 
к "К         ╓Т
5__inference_stream_0_input_drop_layer_call_fn_5262826Y8в5
.в+
%К"
inputs         ╓
p
к "К         ╓