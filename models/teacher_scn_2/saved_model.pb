²
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
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
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48�
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
�
Adam/v/teacher_outputs/biasVarHandleOp*
_output_shapes
: *,

debug_nameAdam/v/teacher_outputs/bias/*
dtype0*
shape:*,
shared_nameAdam/v/teacher_outputs/bias
�
/Adam/v/teacher_outputs/bias/Read/ReadVariableOpReadVariableOpAdam/v/teacher_outputs/bias*
_output_shapes
:*
dtype0
�
Adam/m/teacher_outputs/biasVarHandleOp*
_output_shapes
: *,

debug_nameAdam/m/teacher_outputs/bias/*
dtype0*
shape:*,
shared_nameAdam/m/teacher_outputs/bias
�
/Adam/m/teacher_outputs/bias/Read/ReadVariableOpReadVariableOpAdam/m/teacher_outputs/bias*
_output_shapes
:*
dtype0
�
Adam/v/teacher_outputs/kernelVarHandleOp*
_output_shapes
: *.

debug_name Adam/v/teacher_outputs/kernel/*
dtype0*
shape:*.
shared_nameAdam/v/teacher_outputs/kernel
�
1Adam/v/teacher_outputs/kernel/Read/ReadVariableOpReadVariableOpAdam/v/teacher_outputs/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/teacher_outputs/kernelVarHandleOp*
_output_shapes
: *.

debug_name Adam/m/teacher_outputs/kernel/*
dtype0*
shape:*.
shared_nameAdam/m/teacher_outputs/kernel
�
1Adam/m/teacher_outputs/kernel/Read/ReadVariableOpReadVariableOpAdam/m/teacher_outputs/kernel*&
_output_shapes
:*
dtype0
�
Adam/v/teacher_conv2d_4/biasVarHandleOp*
_output_shapes
: *-

debug_nameAdam/v/teacher_conv2d_4/bias/*
dtype0*
shape:*-
shared_nameAdam/v/teacher_conv2d_4/bias
�
0Adam/v/teacher_conv2d_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/teacher_conv2d_4/bias*
_output_shapes
:*
dtype0
�
Adam/m/teacher_conv2d_4/biasVarHandleOp*
_output_shapes
: *-

debug_nameAdam/m/teacher_conv2d_4/bias/*
dtype0*
shape:*-
shared_nameAdam/m/teacher_conv2d_4/bias
�
0Adam/m/teacher_conv2d_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/teacher_conv2d_4/bias*
_output_shapes
:*
dtype0
�
Adam/v/teacher_conv2d_4/kernelVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/teacher_conv2d_4/kernel/*
dtype0*
shape:*/
shared_name Adam/v/teacher_conv2d_4/kernel
�
2Adam/v/teacher_conv2d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/teacher_conv2d_4/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/teacher_conv2d_4/kernelVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/teacher_conv2d_4/kernel/*
dtype0*
shape:*/
shared_name Adam/m/teacher_conv2d_4/kernel
�
2Adam/m/teacher_conv2d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/teacher_conv2d_4/kernel*&
_output_shapes
:*
dtype0
�
Adam/v/teacher_conv2d_3/biasVarHandleOp*
_output_shapes
: *-

debug_nameAdam/v/teacher_conv2d_3/bias/*
dtype0*
shape:*-
shared_nameAdam/v/teacher_conv2d_3/bias
�
0Adam/v/teacher_conv2d_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/teacher_conv2d_3/bias*
_output_shapes
:*
dtype0
�
Adam/m/teacher_conv2d_3/biasVarHandleOp*
_output_shapes
: *-

debug_nameAdam/m/teacher_conv2d_3/bias/*
dtype0*
shape:*-
shared_nameAdam/m/teacher_conv2d_3/bias
�
0Adam/m/teacher_conv2d_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/teacher_conv2d_3/bias*
_output_shapes
:*
dtype0
�
Adam/v/teacher_conv2d_3/kernelVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/teacher_conv2d_3/kernel/*
dtype0*
shape:*/
shared_name Adam/v/teacher_conv2d_3/kernel
�
2Adam/v/teacher_conv2d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/teacher_conv2d_3/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/teacher_conv2d_3/kernelVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/teacher_conv2d_3/kernel/*
dtype0*
shape:*/
shared_name Adam/m/teacher_conv2d_3/kernel
�
2Adam/m/teacher_conv2d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/teacher_conv2d_3/kernel*&
_output_shapes
:*
dtype0
�
Adam/v/teacher_dense/biasVarHandleOp*
_output_shapes
: **

debug_nameAdam/v/teacher_dense/bias/*
dtype0*
shape:�**
shared_nameAdam/v/teacher_dense/bias
�
-Adam/v/teacher_dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/teacher_dense/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/teacher_dense/biasVarHandleOp*
_output_shapes
: **

debug_nameAdam/m/teacher_dense/bias/*
dtype0*
shape:�**
shared_nameAdam/m/teacher_dense/bias
�
-Adam/m/teacher_dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/teacher_dense/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/teacher_dense/kernelVarHandleOp*
_output_shapes
: *,

debug_nameAdam/v/teacher_dense/kernel/*
dtype0*
shape:	P�*,
shared_nameAdam/v/teacher_dense/kernel
�
/Adam/v/teacher_dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/teacher_dense/kernel*
_output_shapes
:	P�*
dtype0
�
Adam/m/teacher_dense/kernelVarHandleOp*
_output_shapes
: *,

debug_nameAdam/m/teacher_dense/kernel/*
dtype0*
shape:	P�*,
shared_nameAdam/m/teacher_dense/kernel
�
/Adam/m/teacher_dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/teacher_dense/kernel*
_output_shapes
:	P�*
dtype0
�
Adam/v/teacher_latent/biasVarHandleOp*
_output_shapes
: *+

debug_nameAdam/v/teacher_latent/bias/*
dtype0*
shape:P*+
shared_nameAdam/v/teacher_latent/bias
�
.Adam/v/teacher_latent/bias/Read/ReadVariableOpReadVariableOpAdam/v/teacher_latent/bias*
_output_shapes
:P*
dtype0
�
Adam/m/teacher_latent/biasVarHandleOp*
_output_shapes
: *+

debug_nameAdam/m/teacher_latent/bias/*
dtype0*
shape:P*+
shared_nameAdam/m/teacher_latent/bias
�
.Adam/m/teacher_latent/bias/Read/ReadVariableOpReadVariableOpAdam/m/teacher_latent/bias*
_output_shapes
:P*
dtype0
�
Adam/v/teacher_latent/kernelVarHandleOp*
_output_shapes
: *-

debug_nameAdam/v/teacher_latent/kernel/*
dtype0*
shape:	�P*-
shared_nameAdam/v/teacher_latent/kernel
�
0Adam/v/teacher_latent/kernel/Read/ReadVariableOpReadVariableOpAdam/v/teacher_latent/kernel*
_output_shapes
:	�P*
dtype0
�
Adam/m/teacher_latent/kernelVarHandleOp*
_output_shapes
: *-

debug_nameAdam/m/teacher_latent/kernel/*
dtype0*
shape:	�P*-
shared_nameAdam/m/teacher_latent/kernel
�
0Adam/m/teacher_latent/kernel/Read/ReadVariableOpReadVariableOpAdam/m/teacher_latent/kernel*
_output_shapes
:	�P*
dtype0
�
Adam/v/teacher_conv2d_2/biasVarHandleOp*
_output_shapes
: *-

debug_nameAdam/v/teacher_conv2d_2/bias/*
dtype0*
shape:*-
shared_nameAdam/v/teacher_conv2d_2/bias
�
0Adam/v/teacher_conv2d_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/teacher_conv2d_2/bias*
_output_shapes
:*
dtype0
�
Adam/m/teacher_conv2d_2/biasVarHandleOp*
_output_shapes
: *-

debug_nameAdam/m/teacher_conv2d_2/bias/*
dtype0*
shape:*-
shared_nameAdam/m/teacher_conv2d_2/bias
�
0Adam/m/teacher_conv2d_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/teacher_conv2d_2/bias*
_output_shapes
:*
dtype0
�
Adam/v/teacher_conv2d_2/kernelVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/teacher_conv2d_2/kernel/*
dtype0*
shape:*/
shared_name Adam/v/teacher_conv2d_2/kernel
�
2Adam/v/teacher_conv2d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/teacher_conv2d_2/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/teacher_conv2d_2/kernelVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/teacher_conv2d_2/kernel/*
dtype0*
shape:*/
shared_name Adam/m/teacher_conv2d_2/kernel
�
2Adam/m/teacher_conv2d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/teacher_conv2d_2/kernel*&
_output_shapes
:*
dtype0
�
Adam/v/teacher_conv2d_1/biasVarHandleOp*
_output_shapes
: *-

debug_nameAdam/v/teacher_conv2d_1/bias/*
dtype0*
shape:*-
shared_nameAdam/v/teacher_conv2d_1/bias
�
0Adam/v/teacher_conv2d_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/teacher_conv2d_1/bias*
_output_shapes
:*
dtype0
�
Adam/m/teacher_conv2d_1/biasVarHandleOp*
_output_shapes
: *-

debug_nameAdam/m/teacher_conv2d_1/bias/*
dtype0*
shape:*-
shared_nameAdam/m/teacher_conv2d_1/bias
�
0Adam/m/teacher_conv2d_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/teacher_conv2d_1/bias*
_output_shapes
:*
dtype0
�
Adam/v/teacher_conv2d_1/kernelVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/teacher_conv2d_1/kernel/*
dtype0*
shape:*/
shared_name Adam/v/teacher_conv2d_1/kernel
�
2Adam/v/teacher_conv2d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/teacher_conv2d_1/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/teacher_conv2d_1/kernelVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/teacher_conv2d_1/kernel/*
dtype0*
shape:*/
shared_name Adam/m/teacher_conv2d_1/kernel
�
2Adam/m/teacher_conv2d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/teacher_conv2d_1/kernel*&
_output_shapes
:*
dtype0
�
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
teacher_outputs/biasVarHandleOp*
_output_shapes
: *%

debug_nameteacher_outputs/bias/*
dtype0*
shape:*%
shared_nameteacher_outputs/bias
y
(teacher_outputs/bias/Read/ReadVariableOpReadVariableOpteacher_outputs/bias*
_output_shapes
:*
dtype0
�
teacher_outputs/kernelVarHandleOp*
_output_shapes
: *'

debug_nameteacher_outputs/kernel/*
dtype0*
shape:*'
shared_nameteacher_outputs/kernel
�
*teacher_outputs/kernel/Read/ReadVariableOpReadVariableOpteacher_outputs/kernel*&
_output_shapes
:*
dtype0
�
teacher_conv2d_4/biasVarHandleOp*
_output_shapes
: *&

debug_nameteacher_conv2d_4/bias/*
dtype0*
shape:*&
shared_nameteacher_conv2d_4/bias
{
)teacher_conv2d_4/bias/Read/ReadVariableOpReadVariableOpteacher_conv2d_4/bias*
_output_shapes
:*
dtype0
�
teacher_conv2d_4/kernelVarHandleOp*
_output_shapes
: *(

debug_nameteacher_conv2d_4/kernel/*
dtype0*
shape:*(
shared_nameteacher_conv2d_4/kernel
�
+teacher_conv2d_4/kernel/Read/ReadVariableOpReadVariableOpteacher_conv2d_4/kernel*&
_output_shapes
:*
dtype0
�
teacher_conv2d_3/biasVarHandleOp*
_output_shapes
: *&

debug_nameteacher_conv2d_3/bias/*
dtype0*
shape:*&
shared_nameteacher_conv2d_3/bias
{
)teacher_conv2d_3/bias/Read/ReadVariableOpReadVariableOpteacher_conv2d_3/bias*
_output_shapes
:*
dtype0
�
teacher_conv2d_3/kernelVarHandleOp*
_output_shapes
: *(

debug_nameteacher_conv2d_3/kernel/*
dtype0*
shape:*(
shared_nameteacher_conv2d_3/kernel
�
+teacher_conv2d_3/kernel/Read/ReadVariableOpReadVariableOpteacher_conv2d_3/kernel*&
_output_shapes
:*
dtype0
�
teacher_dense/biasVarHandleOp*
_output_shapes
: *#

debug_nameteacher_dense/bias/*
dtype0*
shape:�*#
shared_nameteacher_dense/bias
v
&teacher_dense/bias/Read/ReadVariableOpReadVariableOpteacher_dense/bias*
_output_shapes	
:�*
dtype0
�
teacher_dense/kernelVarHandleOp*
_output_shapes
: *%

debug_nameteacher_dense/kernel/*
dtype0*
shape:	P�*%
shared_nameteacher_dense/kernel
~
(teacher_dense/kernel/Read/ReadVariableOpReadVariableOpteacher_dense/kernel*
_output_shapes
:	P�*
dtype0
�
teacher_latent/biasVarHandleOp*
_output_shapes
: *$

debug_nameteacher_latent/bias/*
dtype0*
shape:P*$
shared_nameteacher_latent/bias
w
'teacher_latent/bias/Read/ReadVariableOpReadVariableOpteacher_latent/bias*
_output_shapes
:P*
dtype0
�
teacher_latent/kernelVarHandleOp*
_output_shapes
: *&

debug_nameteacher_latent/kernel/*
dtype0*
shape:	�P*&
shared_nameteacher_latent/kernel
�
)teacher_latent/kernel/Read/ReadVariableOpReadVariableOpteacher_latent/kernel*
_output_shapes
:	�P*
dtype0
�
teacher_conv2d_2/biasVarHandleOp*
_output_shapes
: *&

debug_nameteacher_conv2d_2/bias/*
dtype0*
shape:*&
shared_nameteacher_conv2d_2/bias
{
)teacher_conv2d_2/bias/Read/ReadVariableOpReadVariableOpteacher_conv2d_2/bias*
_output_shapes
:*
dtype0
�
teacher_conv2d_2/kernelVarHandleOp*
_output_shapes
: *(

debug_nameteacher_conv2d_2/kernel/*
dtype0*
shape:*(
shared_nameteacher_conv2d_2/kernel
�
+teacher_conv2d_2/kernel/Read/ReadVariableOpReadVariableOpteacher_conv2d_2/kernel*&
_output_shapes
:*
dtype0
�
teacher_conv2d_1/biasVarHandleOp*
_output_shapes
: *&

debug_nameteacher_conv2d_1/bias/*
dtype0*
shape:*&
shared_nameteacher_conv2d_1/bias
{
)teacher_conv2d_1/bias/Read/ReadVariableOpReadVariableOpteacher_conv2d_1/bias*
_output_shapes
:*
dtype0
�
teacher_conv2d_1/kernelVarHandleOp*
_output_shapes
: *(

debug_nameteacher_conv2d_1/kernel/*
dtype0*
shape:*(
shared_nameteacher_conv2d_1/kernel
�
+teacher_conv2d_1/kernel/Read/ReadVariableOpReadVariableOpteacher_conv2d_1/kernel*&
_output_shapes
:*
dtype0
�
serving_default_teacher_inputs_Placeholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_teacher_inputs_teacher_conv2d_1/kernelteacher_conv2d_1/biasteacher_conv2d_2/kernelteacher_conv2d_2/biasteacher_latent/kernelteacher_latent/biasteacher_dense/kernelteacher_dense/biasteacher_conv2d_3/kernelteacher_conv2d_3/biasteacher_conv2d_4/kernelteacher_conv2d_4/biasteacher_outputs/kernelteacher_outputs/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_116055

NoOpNoOp
�~
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�~
value�~B�~ B�}
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer_with_weights-4
layer-12
layer-13
layer-14
layer_with_weights-5
layer-15
layer-16
layer_with_weights-6
layer-17
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses* 
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias
 *_jit_compiled_convolution_op*
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses* 
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses* 
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias
 ?_jit_compiled_convolution_op*
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses* 
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses* 
�
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses

Rkernel
Sbias*
�
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

Zkernel
[bias*
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses* 
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses* 
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses

nkernel
obias
 p_jit_compiled_convolution_op*
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses* 
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses* 
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
n
(0
)1
=2
>3
R4
S5
Z6
[7
n8
o9
�10
�11
�12
�13*
n
(0
)1
=2
>3
R4
S5
Z6
[7
n8
o9
�10
�11
�12
�13*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

(0
)1*

(0
)1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ga
VARIABLE_VALUEteacher_conv2d_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEteacher_conv2d_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

=0
>1*

=0
>1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ga
VARIABLE_VALUEteacher_conv2d_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEteacher_conv2d_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

R0
S1*

R0
S1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEteacher_latent/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEteacher_latent/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

Z0
[1*

Z0
[1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
d^
VARIABLE_VALUEteacher_dense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEteacher_dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

n0
o1*

n0
o1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ga
VARIABLE_VALUEteacher_conv2d_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEteacher_conv2d_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ga
VARIABLE_VALUEteacher_conv2d_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEteacher_conv2d_4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
f`
VARIABLE_VALUEteacher_outputs/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEteacher_outputs/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17*

�0*
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
x
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13*
x
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
ic
VARIABLE_VALUEAdam/m/teacher_conv2d_1/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/teacher_conv2d_1/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/m/teacher_conv2d_1/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/v/teacher_conv2d_1/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/teacher_conv2d_2/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/teacher_conv2d_2/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/m/teacher_conv2d_2/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/v/teacher_conv2d_2/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/m/teacher_latent/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/teacher_latent/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/teacher_latent/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/teacher_latent/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/m/teacher_dense/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/v/teacher_dense/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/teacher_dense/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/teacher_dense/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/teacher_conv2d_3/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/teacher_conv2d_3/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/m/teacher_conv2d_3/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/teacher_conv2d_3/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/teacher_conv2d_4/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/teacher_conv2d_4/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/m/teacher_conv2d_4/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/teacher_conv2d_4/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/teacher_outputs/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/teacher_outputs/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/m/teacher_outputs/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/v/teacher_outputs/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameteacher_conv2d_1/kernelteacher_conv2d_1/biasteacher_conv2d_2/kernelteacher_conv2d_2/biasteacher_latent/kernelteacher_latent/biasteacher_dense/kernelteacher_dense/biasteacher_conv2d_3/kernelteacher_conv2d_3/biasteacher_conv2d_4/kernelteacher_conv2d_4/biasteacher_outputs/kernelteacher_outputs/bias	iterationlearning_rateAdam/m/teacher_conv2d_1/kernelAdam/v/teacher_conv2d_1/kernelAdam/m/teacher_conv2d_1/biasAdam/v/teacher_conv2d_1/biasAdam/m/teacher_conv2d_2/kernelAdam/v/teacher_conv2d_2/kernelAdam/m/teacher_conv2d_2/biasAdam/v/teacher_conv2d_2/biasAdam/m/teacher_latent/kernelAdam/v/teacher_latent/kernelAdam/m/teacher_latent/biasAdam/v/teacher_latent/biasAdam/m/teacher_dense/kernelAdam/v/teacher_dense/kernelAdam/m/teacher_dense/biasAdam/v/teacher_dense/biasAdam/m/teacher_conv2d_3/kernelAdam/v/teacher_conv2d_3/kernelAdam/m/teacher_conv2d_3/biasAdam/v/teacher_conv2d_3/biasAdam/m/teacher_conv2d_4/kernelAdam/v/teacher_conv2d_4/kernelAdam/m/teacher_conv2d_4/biasAdam/v/teacher_conv2d_4/biasAdam/m/teacher_outputs/kernelAdam/v/teacher_outputs/kernelAdam/m/teacher_outputs/biasAdam/v/teacher_outputs/biastotalcountConst*;
Tin4
220*
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
__inference__traced_save_116614
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameteacher_conv2d_1/kernelteacher_conv2d_1/biasteacher_conv2d_2/kernelteacher_conv2d_2/biasteacher_latent/kernelteacher_latent/biasteacher_dense/kernelteacher_dense/biasteacher_conv2d_3/kernelteacher_conv2d_3/biasteacher_conv2d_4/kernelteacher_conv2d_4/biasteacher_outputs/kernelteacher_outputs/bias	iterationlearning_rateAdam/m/teacher_conv2d_1/kernelAdam/v/teacher_conv2d_1/kernelAdam/m/teacher_conv2d_1/biasAdam/v/teacher_conv2d_1/biasAdam/m/teacher_conv2d_2/kernelAdam/v/teacher_conv2d_2/kernelAdam/m/teacher_conv2d_2/biasAdam/v/teacher_conv2d_2/biasAdam/m/teacher_latent/kernelAdam/v/teacher_latent/kernelAdam/m/teacher_latent/biasAdam/v/teacher_latent/biasAdam/m/teacher_dense/kernelAdam/v/teacher_dense/kernelAdam/m/teacher_dense/biasAdam/v/teacher_dense/biasAdam/m/teacher_conv2d_3/kernelAdam/v/teacher_conv2d_3/kernelAdam/m/teacher_conv2d_3/biasAdam/v/teacher_conv2d_3/biasAdam/m/teacher_conv2d_4/kernelAdam/v/teacher_conv2d_4/kernelAdam/m/teacher_conv2d_4/biasAdam/v/teacher_conv2d_4/biasAdam/m/teacher_outputs/kernelAdam/v/teacher_outputs/kernelAdam/m/teacher_outputs/biasAdam/v/teacher_outputs/biastotalcount*:
Tin3
12/*
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
"__inference__traced_restore_116761̢

�
�
$__inference_signature_wrapper_116055
teacher_inputs_!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	�P
	unknown_4:P
	unknown_5:	P�
	unknown_6:	�#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallteacher_inputs_unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_115594w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name116051:&"
 
_user_specified_name116049:&"
 
_user_specified_name116047:&"
 
_user_specified_name116045:&
"
 
_user_specified_name116043:&	"
 
_user_specified_name116041:&"
 
_user_specified_name116039:&"
 
_user_specified_name116037:&"
 
_user_specified_name116035:&"
 
_user_specified_name116033:&"
 
_user_specified_name116031:&"
 
_user_specified_name116029:&"
 
_user_specified_name116027:&"
 
_user_specified_name116025:` \
/
_output_shapes
:���������
)
_user_specified_nameteacher_inputs_
�
f
J__inference_teacher_relu_3_layer_call_and_return_conditional_losses_115739

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
L__inference_teacher_conv2d_3_layer_call_and_return_conditional_losses_116240

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
M
1__inference_teacher_reshape2_layer_call_fn_116197

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_teacher_reshape2_layer_call_and_return_conditional_losses_115733h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
j
N__inference_teacher_upsampling_layer_call_and_return_conditional_losses_115616

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
L__inference_teacher_conv2d_2_layer_call_and_return_conditional_losses_116132

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�,
__inference__traced_save_116614
file_prefixH
.read_disablecopyonread_teacher_conv2d_1_kernel:<
.read_1_disablecopyonread_teacher_conv2d_1_bias:J
0read_2_disablecopyonread_teacher_conv2d_2_kernel:<
.read_3_disablecopyonread_teacher_conv2d_2_bias:A
.read_4_disablecopyonread_teacher_latent_kernel:	�P:
,read_5_disablecopyonread_teacher_latent_bias:P@
-read_6_disablecopyonread_teacher_dense_kernel:	P�:
+read_7_disablecopyonread_teacher_dense_bias:	�J
0read_8_disablecopyonread_teacher_conv2d_3_kernel:<
.read_9_disablecopyonread_teacher_conv2d_3_bias:K
1read_10_disablecopyonread_teacher_conv2d_4_kernel:=
/read_11_disablecopyonread_teacher_conv2d_4_bias:J
0read_12_disablecopyonread_teacher_outputs_kernel:<
.read_13_disablecopyonread_teacher_outputs_bias:-
#read_14_disablecopyonread_iteration:	 1
'read_15_disablecopyonread_learning_rate: R
8read_16_disablecopyonread_adam_m_teacher_conv2d_1_kernel:R
8read_17_disablecopyonread_adam_v_teacher_conv2d_1_kernel:D
6read_18_disablecopyonread_adam_m_teacher_conv2d_1_bias:D
6read_19_disablecopyonread_adam_v_teacher_conv2d_1_bias:R
8read_20_disablecopyonread_adam_m_teacher_conv2d_2_kernel:R
8read_21_disablecopyonread_adam_v_teacher_conv2d_2_kernel:D
6read_22_disablecopyonread_adam_m_teacher_conv2d_2_bias:D
6read_23_disablecopyonread_adam_v_teacher_conv2d_2_bias:I
6read_24_disablecopyonread_adam_m_teacher_latent_kernel:	�PI
6read_25_disablecopyonread_adam_v_teacher_latent_kernel:	�PB
4read_26_disablecopyonread_adam_m_teacher_latent_bias:PB
4read_27_disablecopyonread_adam_v_teacher_latent_bias:PH
5read_28_disablecopyonread_adam_m_teacher_dense_kernel:	P�H
5read_29_disablecopyonread_adam_v_teacher_dense_kernel:	P�B
3read_30_disablecopyonread_adam_m_teacher_dense_bias:	�B
3read_31_disablecopyonread_adam_v_teacher_dense_bias:	�R
8read_32_disablecopyonread_adam_m_teacher_conv2d_3_kernel:R
8read_33_disablecopyonread_adam_v_teacher_conv2d_3_kernel:D
6read_34_disablecopyonread_adam_m_teacher_conv2d_3_bias:D
6read_35_disablecopyonread_adam_v_teacher_conv2d_3_bias:R
8read_36_disablecopyonread_adam_m_teacher_conv2d_4_kernel:R
8read_37_disablecopyonread_adam_v_teacher_conv2d_4_kernel:D
6read_38_disablecopyonread_adam_m_teacher_conv2d_4_bias:D
6read_39_disablecopyonread_adam_v_teacher_conv2d_4_bias:Q
7read_40_disablecopyonread_adam_m_teacher_outputs_kernel:Q
7read_41_disablecopyonread_adam_v_teacher_outputs_kernel:C
5read_42_disablecopyonread_adam_m_teacher_outputs_bias:C
5read_43_disablecopyonread_adam_v_teacher_outputs_bias:)
read_44_disablecopyonread_total: )
read_45_disablecopyonread_count: 
savev2_const
identity_93��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
Read/DisableCopyOnReadDisableCopyOnRead.read_disablecopyonread_teacher_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp.read_disablecopyonread_teacher_conv2d_1_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_1/DisableCopyOnReadDisableCopyOnRead.read_1_disablecopyonread_teacher_conv2d_1_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp.read_1_disablecopyonread_teacher_conv2d_1_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_2/DisableCopyOnReadDisableCopyOnRead0read_2_disablecopyonread_teacher_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp0read_2_disablecopyonread_teacher_conv2d_2_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_3/DisableCopyOnReadDisableCopyOnRead.read_3_disablecopyonread_teacher_conv2d_2_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp.read_3_disablecopyonread_teacher_conv2d_2_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_4/DisableCopyOnReadDisableCopyOnRead.read_4_disablecopyonread_teacher_latent_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp.read_4_disablecopyonread_teacher_latent_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�P*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�Pd

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	�P�
Read_5/DisableCopyOnReadDisableCopyOnRead,read_5_disablecopyonread_teacher_latent_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp,read_5_disablecopyonread_teacher_latent_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:P*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Pa
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:P�
Read_6/DisableCopyOnReadDisableCopyOnRead-read_6_disablecopyonread_teacher_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp-read_6_disablecopyonread_teacher_dense_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	P�*
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	P�f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	P�
Read_7/DisableCopyOnReadDisableCopyOnRead+read_7_disablecopyonread_teacher_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp+read_7_disablecopyonread_teacher_dense_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_8/DisableCopyOnReadDisableCopyOnRead0read_8_disablecopyonread_teacher_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp0read_8_disablecopyonread_teacher_conv2d_3_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0v
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_9/DisableCopyOnReadDisableCopyOnRead.read_9_disablecopyonread_teacher_conv2d_3_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp.read_9_disablecopyonread_teacher_conv2d_3_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_10/DisableCopyOnReadDisableCopyOnRead1read_10_disablecopyonread_teacher_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp1read_10_disablecopyonread_teacher_conv2d_4_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_11/DisableCopyOnReadDisableCopyOnRead/read_11_disablecopyonread_teacher_conv2d_4_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp/read_11_disablecopyonread_teacher_conv2d_4_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_12/DisableCopyOnReadDisableCopyOnRead0read_12_disablecopyonread_teacher_outputs_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp0read_12_disablecopyonread_teacher_outputs_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_13/DisableCopyOnReadDisableCopyOnRead.read_13_disablecopyonread_teacher_outputs_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp.read_13_disablecopyonread_teacher_outputs_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_14/DisableCopyOnReadDisableCopyOnRead#read_14_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp#read_14_disablecopyonread_iteration^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_15/DisableCopyOnReadDisableCopyOnRead'read_15_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp'read_15_disablecopyonread_learning_rate^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_16/DisableCopyOnReadDisableCopyOnRead8read_16_disablecopyonread_adam_m_teacher_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp8read_16_disablecopyonread_adam_m_teacher_conv2d_1_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_17/DisableCopyOnReadDisableCopyOnRead8read_17_disablecopyonread_adam_v_teacher_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp8read_17_disablecopyonread_adam_v_teacher_conv2d_1_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_18/DisableCopyOnReadDisableCopyOnRead6read_18_disablecopyonread_adam_m_teacher_conv2d_1_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp6read_18_disablecopyonread_adam_m_teacher_conv2d_1_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_19/DisableCopyOnReadDisableCopyOnRead6read_19_disablecopyonread_adam_v_teacher_conv2d_1_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp6read_19_disablecopyonread_adam_v_teacher_conv2d_1_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_20/DisableCopyOnReadDisableCopyOnRead8read_20_disablecopyonread_adam_m_teacher_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp8read_20_disablecopyonread_adam_m_teacher_conv2d_2_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_21/DisableCopyOnReadDisableCopyOnRead8read_21_disablecopyonread_adam_v_teacher_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp8read_21_disablecopyonread_adam_v_teacher_conv2d_2_kernel^Read_21/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_22/DisableCopyOnReadDisableCopyOnRead6read_22_disablecopyonread_adam_m_teacher_conv2d_2_bias"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp6read_22_disablecopyonread_adam_m_teacher_conv2d_2_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_23/DisableCopyOnReadDisableCopyOnRead6read_23_disablecopyonread_adam_v_teacher_conv2d_2_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp6read_23_disablecopyonread_adam_v_teacher_conv2d_2_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_24/DisableCopyOnReadDisableCopyOnRead6read_24_disablecopyonread_adam_m_teacher_latent_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp6read_24_disablecopyonread_adam_m_teacher_latent_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�P*
dtype0p
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�Pf
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:	�P�
Read_25/DisableCopyOnReadDisableCopyOnRead6read_25_disablecopyonread_adam_v_teacher_latent_kernel"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp6read_25_disablecopyonread_adam_v_teacher_latent_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�P*
dtype0p
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�Pf
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:	�P�
Read_26/DisableCopyOnReadDisableCopyOnRead4read_26_disablecopyonread_adam_m_teacher_latent_bias"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp4read_26_disablecopyonread_adam_m_teacher_latent_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:P*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Pa
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:P�
Read_27/DisableCopyOnReadDisableCopyOnRead4read_27_disablecopyonread_adam_v_teacher_latent_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp4read_27_disablecopyonread_adam_v_teacher_latent_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:P*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Pa
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:P�
Read_28/DisableCopyOnReadDisableCopyOnRead5read_28_disablecopyonread_adam_m_teacher_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp5read_28_disablecopyonread_adam_m_teacher_dense_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	P�*
dtype0p
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	P�f
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:	P��
Read_29/DisableCopyOnReadDisableCopyOnRead5read_29_disablecopyonread_adam_v_teacher_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp5read_29_disablecopyonread_adam_v_teacher_dense_kernel^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	P�*
dtype0p
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	P�f
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:	P��
Read_30/DisableCopyOnReadDisableCopyOnRead3read_30_disablecopyonread_adam_m_teacher_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp3read_30_disablecopyonread_adam_m_teacher_dense_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_31/DisableCopyOnReadDisableCopyOnRead3read_31_disablecopyonread_adam_v_teacher_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp3read_31_disablecopyonread_adam_v_teacher_dense_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_32/DisableCopyOnReadDisableCopyOnRead8read_32_disablecopyonread_adam_m_teacher_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp8read_32_disablecopyonread_adam_m_teacher_conv2d_3_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_33/DisableCopyOnReadDisableCopyOnRead8read_33_disablecopyonread_adam_v_teacher_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp8read_33_disablecopyonread_adam_v_teacher_conv2d_3_kernel^Read_33/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_34/DisableCopyOnReadDisableCopyOnRead6read_34_disablecopyonread_adam_m_teacher_conv2d_3_bias"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp6read_34_disablecopyonread_adam_m_teacher_conv2d_3_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_35/DisableCopyOnReadDisableCopyOnRead6read_35_disablecopyonread_adam_v_teacher_conv2d_3_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp6read_35_disablecopyonread_adam_v_teacher_conv2d_3_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_36/DisableCopyOnReadDisableCopyOnRead8read_36_disablecopyonread_adam_m_teacher_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp8read_36_disablecopyonread_adam_m_teacher_conv2d_4_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_37/DisableCopyOnReadDisableCopyOnRead8read_37_disablecopyonread_adam_v_teacher_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp8read_37_disablecopyonread_adam_v_teacher_conv2d_4_kernel^Read_37/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_38/DisableCopyOnReadDisableCopyOnRead6read_38_disablecopyonread_adam_m_teacher_conv2d_4_bias"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp6read_38_disablecopyonread_adam_m_teacher_conv2d_4_bias^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_39/DisableCopyOnReadDisableCopyOnRead6read_39_disablecopyonread_adam_v_teacher_conv2d_4_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp6read_39_disablecopyonread_adam_v_teacher_conv2d_4_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_40/DisableCopyOnReadDisableCopyOnRead7read_40_disablecopyonread_adam_m_teacher_outputs_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp7read_40_disablecopyonread_adam_m_teacher_outputs_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_41/DisableCopyOnReadDisableCopyOnRead7read_41_disablecopyonread_adam_v_teacher_outputs_kernel"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp7read_41_disablecopyonread_adam_v_teacher_outputs_kernel^Read_41/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_42/DisableCopyOnReadDisableCopyOnRead5read_42_disablecopyonread_adam_m_teacher_outputs_bias"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp5read_42_disablecopyonread_adam_m_teacher_outputs_bias^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_43/DisableCopyOnReadDisableCopyOnRead5read_43_disablecopyonread_adam_v_teacher_outputs_bias"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp5read_43_disablecopyonread_adam_v_teacher_outputs_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_44/DisableCopyOnReadDisableCopyOnReadread_44_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOpread_44_disablecopyonread_total^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_45/DisableCopyOnReadDisableCopyOnReadread_45_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOpread_45_disablecopyonread_count^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*�
value�B�/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *=
dtypes3
12/	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_92Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_93IdentityIdentity_92:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_93Identity_93:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=/9

_output_shapes
: 

_user_specified_nameConst:%.!

_user_specified_namecount:%-!

_user_specified_nametotal:;,7
5
_user_specified_nameAdam/v/teacher_outputs/bias:;+7
5
_user_specified_nameAdam/m/teacher_outputs/bias:=*9
7
_user_specified_nameAdam/v/teacher_outputs/kernel:=)9
7
_user_specified_nameAdam/m/teacher_outputs/kernel:<(8
6
_user_specified_nameAdam/v/teacher_conv2d_4/bias:<'8
6
_user_specified_nameAdam/m/teacher_conv2d_4/bias:>&:
8
_user_specified_name Adam/v/teacher_conv2d_4/kernel:>%:
8
_user_specified_name Adam/m/teacher_conv2d_4/kernel:<$8
6
_user_specified_nameAdam/v/teacher_conv2d_3/bias:<#8
6
_user_specified_nameAdam/m/teacher_conv2d_3/bias:>":
8
_user_specified_name Adam/v/teacher_conv2d_3/kernel:>!:
8
_user_specified_name Adam/m/teacher_conv2d_3/kernel:9 5
3
_user_specified_nameAdam/v/teacher_dense/bias:95
3
_user_specified_nameAdam/m/teacher_dense/bias:;7
5
_user_specified_nameAdam/v/teacher_dense/kernel:;7
5
_user_specified_nameAdam/m/teacher_dense/kernel::6
4
_user_specified_nameAdam/v/teacher_latent/bias::6
4
_user_specified_nameAdam/m/teacher_latent/bias:<8
6
_user_specified_nameAdam/v/teacher_latent/kernel:<8
6
_user_specified_nameAdam/m/teacher_latent/kernel:<8
6
_user_specified_nameAdam/v/teacher_conv2d_2/bias:<8
6
_user_specified_nameAdam/m/teacher_conv2d_2/bias:>:
8
_user_specified_name Adam/v/teacher_conv2d_2/kernel:>:
8
_user_specified_name Adam/m/teacher_conv2d_2/kernel:<8
6
_user_specified_nameAdam/v/teacher_conv2d_1/bias:<8
6
_user_specified_nameAdam/m/teacher_conv2d_1/bias:>:
8
_user_specified_name Adam/v/teacher_conv2d_1/kernel:>:
8
_user_specified_name Adam/m/teacher_conv2d_1/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:40
.
_user_specified_nameteacher_outputs/bias:62
0
_user_specified_nameteacher_outputs/kernel:51
/
_user_specified_nameteacher_conv2d_4/bias:73
1
_user_specified_nameteacher_conv2d_4/kernel:5
1
/
_user_specified_nameteacher_conv2d_3/bias:7	3
1
_user_specified_nameteacher_conv2d_3/kernel:2.
,
_user_specified_nameteacher_dense/bias:40
.
_user_specified_nameteacher_dense/kernel:3/
-
_user_specified_nameteacher_latent/bias:51
/
_user_specified_nameteacher_latent/kernel:51
/
_user_specified_nameteacher_conv2d_2/bias:73
1
_user_specified_nameteacher_conv2d_2/kernel:51
/
_user_specified_nameteacher_conv2d_1/bias:73
1
_user_specified_nameteacher_conv2d_1/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
L__inference_teacher_conv2d_4_layer_call_and_return_conditional_losses_116286

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�

�
J__inference_teacher_latent_layer_call_and_return_conditional_losses_116173

inputs1
matmul_readvariableop_resource:	�P-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������PP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������Pa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������PS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
L__inference_teacher_conv2d_4_layer_call_and_return_conditional_losses_115772

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�

�
L__inference_teacher_conv2d_1_layer_call_and_return_conditional_losses_115648

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
K
/__inference_teacher_relu_4_layer_call_fn_116245

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_relu_4_layer_call_and_return_conditional_losses_115760h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
J__inference_teacher_pool_1_layer_call_and_return_conditional_losses_115599

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
g
K__inference_teacher_reshape_layer_call_and_return_conditional_losses_115637

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
K
/__inference_teacher_relu_2_layer_call_fn_116137

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_relu_2_layer_call_and_return_conditional_losses_115680h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
1__inference_teacher_conv2d_4_layer_call_fn_116276

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_teacher_conv2d_4_layer_call_and_return_conditional_losses_115772�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name116272:&"
 
_user_specified_name116270:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�

�
I__inference_teacher_dense_layer_call_and_return_conditional_losses_115714

inputs1
matmul_readvariableop_resource:	P�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	P�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
K
/__inference_teacher_pool_1_layer_call_fn_116108

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_pool_1_layer_call_and_return_conditional_losses_115599�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
f
J__inference_teacher_relu_4_layer_call_and_return_conditional_losses_115760

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_teacher_dense_layer_call_fn_116182

inputs
unknown:	P�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_teacher_dense_layer_call_and_return_conditional_losses_115714p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name116178:&"
 
_user_specified_name116176:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
g
K__inference_teacher_flatten_layer_call_and_return_conditional_losses_115687

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����v  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
J__inference_teacher_relu_5_layer_call_and_return_conditional_losses_115782

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+���������������������������t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
h
L__inference_teacher_reshape2_layer_call_and_return_conditional_losses_115733

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
L__inference_teacher_conv2d_2_layer_call_and_return_conditional_losses_115670

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
J__inference_teacher_pool_1_layer_call_and_return_conditional_losses_116113

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
I__inference_teacher_dense_layer_call_and_return_conditional_losses_116192

inputs1
matmul_readvariableop_resource:	P�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	P�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
K
/__inference_teacher_relu_3_layer_call_fn_116216

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_relu_3_layer_call_and_return_conditional_losses_115739h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
L__inference_teacher_conv2d_1_layer_call_and_return_conditional_losses_116093

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
1__inference_teacher_conv2d_3_layer_call_fn_116230

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_teacher_conv2d_3_layer_call_and_return_conditional_losses_115750w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name116226:&"
 
_user_specified_name116224:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
J__inference_teacher_relu_4_layer_call_and_return_conditional_losses_116250

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
O
3__inference_teacher_upsampling_layer_call_fn_116255

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_teacher_upsampling_layer_call_and_return_conditional_losses_115616�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
��
�
!__inference__wrapped_model_115594
teacher_inputs_W
=teacher_scn_2_teacher_conv2d_1_conv2d_readvariableop_resource:L
>teacher_scn_2_teacher_conv2d_1_biasadd_readvariableop_resource:W
=teacher_scn_2_teacher_conv2d_2_conv2d_readvariableop_resource:L
>teacher_scn_2_teacher_conv2d_2_biasadd_readvariableop_resource:N
;teacher_scn_2_teacher_latent_matmul_readvariableop_resource:	�PJ
<teacher_scn_2_teacher_latent_biasadd_readvariableop_resource:PM
:teacher_scn_2_teacher_dense_matmul_readvariableop_resource:	P�J
;teacher_scn_2_teacher_dense_biasadd_readvariableop_resource:	�W
=teacher_scn_2_teacher_conv2d_3_conv2d_readvariableop_resource:L
>teacher_scn_2_teacher_conv2d_3_biasadd_readvariableop_resource:W
=teacher_scn_2_teacher_conv2d_4_conv2d_readvariableop_resource:L
>teacher_scn_2_teacher_conv2d_4_biasadd_readvariableop_resource:V
<teacher_scn_2_teacher_outputs_conv2d_readvariableop_resource:K
=teacher_scn_2_teacher_outputs_biasadd_readvariableop_resource:
identity��5teacher_scn_2/teacher_conv2d_1/BiasAdd/ReadVariableOp�4teacher_scn_2/teacher_conv2d_1/Conv2D/ReadVariableOp�5teacher_scn_2/teacher_conv2d_2/BiasAdd/ReadVariableOp�4teacher_scn_2/teacher_conv2d_2/Conv2D/ReadVariableOp�5teacher_scn_2/teacher_conv2d_3/BiasAdd/ReadVariableOp�4teacher_scn_2/teacher_conv2d_3/Conv2D/ReadVariableOp�5teacher_scn_2/teacher_conv2d_4/BiasAdd/ReadVariableOp�4teacher_scn_2/teacher_conv2d_4/Conv2D/ReadVariableOp�2teacher_scn_2/teacher_dense/BiasAdd/ReadVariableOp�1teacher_scn_2/teacher_dense/MatMul/ReadVariableOp�3teacher_scn_2/teacher_latent/BiasAdd/ReadVariableOp�2teacher_scn_2/teacher_latent/MatMul/ReadVariableOp�4teacher_scn_2/teacher_outputs/BiasAdd/ReadVariableOp�3teacher_scn_2/teacher_outputs/Conv2D/ReadVariableOpp
#teacher_scn_2/teacher_reshape/ShapeShapeteacher_inputs_*
T0*
_output_shapes
::��{
1teacher_scn_2/teacher_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3teacher_scn_2/teacher_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3teacher_scn_2/teacher_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+teacher_scn_2/teacher_reshape/strided_sliceStridedSlice,teacher_scn_2/teacher_reshape/Shape:output:0:teacher_scn_2/teacher_reshape/strided_slice/stack:output:0<teacher_scn_2/teacher_reshape/strided_slice/stack_1:output:0<teacher_scn_2/teacher_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-teacher_scn_2/teacher_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-teacher_scn_2/teacher_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-teacher_scn_2/teacher_reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
+teacher_scn_2/teacher_reshape/Reshape/shapePack4teacher_scn_2/teacher_reshape/strided_slice:output:06teacher_scn_2/teacher_reshape/Reshape/shape/1:output:06teacher_scn_2/teacher_reshape/Reshape/shape/2:output:06teacher_scn_2/teacher_reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
%teacher_scn_2/teacher_reshape/ReshapeReshapeteacher_inputs_4teacher_scn_2/teacher_reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
4teacher_scn_2/teacher_conv2d_1/Conv2D/ReadVariableOpReadVariableOp=teacher_scn_2_teacher_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
%teacher_scn_2/teacher_conv2d_1/Conv2DConv2D.teacher_scn_2/teacher_reshape/Reshape:output:0<teacher_scn_2/teacher_conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
5teacher_scn_2/teacher_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp>teacher_scn_2_teacher_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
&teacher_scn_2/teacher_conv2d_1/BiasAddBiasAdd.teacher_scn_2/teacher_conv2d_1/Conv2D:output:0=teacher_scn_2/teacher_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
!teacher_scn_2/teacher_relu_1/ReluRelu/teacher_scn_2/teacher_conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:����������
$teacher_scn_2/teacher_pool_1/AvgPoolAvgPool/teacher_scn_2/teacher_relu_1/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
�
4teacher_scn_2/teacher_conv2d_2/Conv2D/ReadVariableOpReadVariableOp=teacher_scn_2_teacher_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
%teacher_scn_2/teacher_conv2d_2/Conv2DConv2D-teacher_scn_2/teacher_pool_1/AvgPool:output:0<teacher_scn_2/teacher_conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
5teacher_scn_2/teacher_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp>teacher_scn_2_teacher_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
&teacher_scn_2/teacher_conv2d_2/BiasAddBiasAdd.teacher_scn_2/teacher_conv2d_2/Conv2D:output:0=teacher_scn_2/teacher_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
!teacher_scn_2/teacher_relu_2/ReluRelu/teacher_scn_2/teacher_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������t
#teacher_scn_2/teacher_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����v  �
%teacher_scn_2/teacher_flatten/ReshapeReshape/teacher_scn_2/teacher_relu_2/Relu:activations:0,teacher_scn_2/teacher_flatten/Const:output:0*
T0*(
_output_shapes
:�����������
2teacher_scn_2/teacher_latent/MatMul/ReadVariableOpReadVariableOp;teacher_scn_2_teacher_latent_matmul_readvariableop_resource*
_output_shapes
:	�P*
dtype0�
#teacher_scn_2/teacher_latent/MatMulMatMul.teacher_scn_2/teacher_flatten/Reshape:output:0:teacher_scn_2/teacher_latent/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
3teacher_scn_2/teacher_latent/BiasAdd/ReadVariableOpReadVariableOp<teacher_scn_2_teacher_latent_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
$teacher_scn_2/teacher_latent/BiasAddBiasAdd-teacher_scn_2/teacher_latent/MatMul:product:0;teacher_scn_2/teacher_latent/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
!teacher_scn_2/teacher_latent/ReluRelu-teacher_scn_2/teacher_latent/BiasAdd:output:0*
T0*'
_output_shapes
:���������P�
1teacher_scn_2/teacher_dense/MatMul/ReadVariableOpReadVariableOp:teacher_scn_2_teacher_dense_matmul_readvariableop_resource*
_output_shapes
:	P�*
dtype0�
"teacher_scn_2/teacher_dense/MatMulMatMul/teacher_scn_2/teacher_latent/Relu:activations:09teacher_scn_2/teacher_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2teacher_scn_2/teacher_dense/BiasAdd/ReadVariableOpReadVariableOp;teacher_scn_2_teacher_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#teacher_scn_2/teacher_dense/BiasAddBiasAdd,teacher_scn_2/teacher_dense/MatMul:product:0:teacher_scn_2/teacher_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$teacher_scn_2/teacher_reshape2/ShapeShape,teacher_scn_2/teacher_dense/BiasAdd:output:0*
T0*
_output_shapes
::��|
2teacher_scn_2/teacher_reshape2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4teacher_scn_2/teacher_reshape2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4teacher_scn_2/teacher_reshape2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
,teacher_scn_2/teacher_reshape2/strided_sliceStridedSlice-teacher_scn_2/teacher_reshape2/Shape:output:0;teacher_scn_2/teacher_reshape2/strided_slice/stack:output:0=teacher_scn_2/teacher_reshape2/strided_slice/stack_1:output:0=teacher_scn_2/teacher_reshape2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.teacher_scn_2/teacher_reshape2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :p
.teacher_scn_2/teacher_reshape2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :p
.teacher_scn_2/teacher_reshape2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
,teacher_scn_2/teacher_reshape2/Reshape/shapePack5teacher_scn_2/teacher_reshape2/strided_slice:output:07teacher_scn_2/teacher_reshape2/Reshape/shape/1:output:07teacher_scn_2/teacher_reshape2/Reshape/shape/2:output:07teacher_scn_2/teacher_reshape2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
&teacher_scn_2/teacher_reshape2/ReshapeReshape,teacher_scn_2/teacher_dense/BiasAdd:output:05teacher_scn_2/teacher_reshape2/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
!teacher_scn_2/teacher_relu_3/ReluRelu/teacher_scn_2/teacher_reshape2/Reshape:output:0*
T0*/
_output_shapes
:����������
4teacher_scn_2/teacher_conv2d_3/Conv2D/ReadVariableOpReadVariableOp=teacher_scn_2_teacher_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
%teacher_scn_2/teacher_conv2d_3/Conv2DConv2D/teacher_scn_2/teacher_relu_3/Relu:activations:0<teacher_scn_2/teacher_conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
5teacher_scn_2/teacher_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp>teacher_scn_2_teacher_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
&teacher_scn_2/teacher_conv2d_3/BiasAddBiasAdd.teacher_scn_2/teacher_conv2d_3/Conv2D:output:0=teacher_scn_2/teacher_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
!teacher_scn_2/teacher_relu_4/ReluRelu/teacher_scn_2/teacher_conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������w
&teacher_scn_2/teacher_upsampling/ConstConst*
_output_shapes
:*
dtype0*
valueB"      y
(teacher_scn_2/teacher_upsampling/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
$teacher_scn_2/teacher_upsampling/mulMul/teacher_scn_2/teacher_upsampling/Const:output:01teacher_scn_2/teacher_upsampling/Const_1:output:0*
T0*
_output_shapes
:�
=teacher_scn_2/teacher_upsampling/resize/ResizeNearestNeighborResizeNearestNeighbor/teacher_scn_2/teacher_relu_4/Relu:activations:0(teacher_scn_2/teacher_upsampling/mul:z:0*
T0*/
_output_shapes
:���������*
half_pixel_centers(�
4teacher_scn_2/teacher_conv2d_4/Conv2D/ReadVariableOpReadVariableOp=teacher_scn_2_teacher_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
%teacher_scn_2/teacher_conv2d_4/Conv2DConv2DNteacher_scn_2/teacher_upsampling/resize/ResizeNearestNeighbor:resized_images:0<teacher_scn_2/teacher_conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
5teacher_scn_2/teacher_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp>teacher_scn_2_teacher_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
&teacher_scn_2/teacher_conv2d_4/BiasAddBiasAdd.teacher_scn_2/teacher_conv2d_4/Conv2D:output:0=teacher_scn_2/teacher_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
!teacher_scn_2/teacher_relu_5/ReluRelu/teacher_scn_2/teacher_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:����������
3teacher_scn_2/teacher_outputs/Conv2D/ReadVariableOpReadVariableOp<teacher_scn_2_teacher_outputs_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
$teacher_scn_2/teacher_outputs/Conv2DConv2D/teacher_scn_2/teacher_relu_5/Relu:activations:0;teacher_scn_2/teacher_outputs/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
4teacher_scn_2/teacher_outputs/BiasAdd/ReadVariableOpReadVariableOp=teacher_scn_2_teacher_outputs_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%teacher_scn_2/teacher_outputs/BiasAddBiasAdd-teacher_scn_2/teacher_outputs/Conv2D:output:0<teacher_scn_2/teacher_outputs/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
"teacher_scn_2/teacher_outputs/ReluRelu.teacher_scn_2/teacher_outputs/BiasAdd:output:0*
T0*/
_output_shapes
:����������
IdentityIdentity0teacher_scn_2/teacher_outputs/Relu:activations:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp6^teacher_scn_2/teacher_conv2d_1/BiasAdd/ReadVariableOp5^teacher_scn_2/teacher_conv2d_1/Conv2D/ReadVariableOp6^teacher_scn_2/teacher_conv2d_2/BiasAdd/ReadVariableOp5^teacher_scn_2/teacher_conv2d_2/Conv2D/ReadVariableOp6^teacher_scn_2/teacher_conv2d_3/BiasAdd/ReadVariableOp5^teacher_scn_2/teacher_conv2d_3/Conv2D/ReadVariableOp6^teacher_scn_2/teacher_conv2d_4/BiasAdd/ReadVariableOp5^teacher_scn_2/teacher_conv2d_4/Conv2D/ReadVariableOp3^teacher_scn_2/teacher_dense/BiasAdd/ReadVariableOp2^teacher_scn_2/teacher_dense/MatMul/ReadVariableOp4^teacher_scn_2/teacher_latent/BiasAdd/ReadVariableOp3^teacher_scn_2/teacher_latent/MatMul/ReadVariableOp5^teacher_scn_2/teacher_outputs/BiasAdd/ReadVariableOp4^teacher_scn_2/teacher_outputs/Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : 2n
5teacher_scn_2/teacher_conv2d_1/BiasAdd/ReadVariableOp5teacher_scn_2/teacher_conv2d_1/BiasAdd/ReadVariableOp2l
4teacher_scn_2/teacher_conv2d_1/Conv2D/ReadVariableOp4teacher_scn_2/teacher_conv2d_1/Conv2D/ReadVariableOp2n
5teacher_scn_2/teacher_conv2d_2/BiasAdd/ReadVariableOp5teacher_scn_2/teacher_conv2d_2/BiasAdd/ReadVariableOp2l
4teacher_scn_2/teacher_conv2d_2/Conv2D/ReadVariableOp4teacher_scn_2/teacher_conv2d_2/Conv2D/ReadVariableOp2n
5teacher_scn_2/teacher_conv2d_3/BiasAdd/ReadVariableOp5teacher_scn_2/teacher_conv2d_3/BiasAdd/ReadVariableOp2l
4teacher_scn_2/teacher_conv2d_3/Conv2D/ReadVariableOp4teacher_scn_2/teacher_conv2d_3/Conv2D/ReadVariableOp2n
5teacher_scn_2/teacher_conv2d_4/BiasAdd/ReadVariableOp5teacher_scn_2/teacher_conv2d_4/BiasAdd/ReadVariableOp2l
4teacher_scn_2/teacher_conv2d_4/Conv2D/ReadVariableOp4teacher_scn_2/teacher_conv2d_4/Conv2D/ReadVariableOp2h
2teacher_scn_2/teacher_dense/BiasAdd/ReadVariableOp2teacher_scn_2/teacher_dense/BiasAdd/ReadVariableOp2f
1teacher_scn_2/teacher_dense/MatMul/ReadVariableOp1teacher_scn_2/teacher_dense/MatMul/ReadVariableOp2j
3teacher_scn_2/teacher_latent/BiasAdd/ReadVariableOp3teacher_scn_2/teacher_latent/BiasAdd/ReadVariableOp2h
2teacher_scn_2/teacher_latent/MatMul/ReadVariableOp2teacher_scn_2/teacher_latent/MatMul/ReadVariableOp2l
4teacher_scn_2/teacher_outputs/BiasAdd/ReadVariableOp4teacher_scn_2/teacher_outputs/BiasAdd/ReadVariableOp2j
3teacher_scn_2/teacher_outputs/Conv2D/ReadVariableOp3teacher_scn_2/teacher_outputs/Conv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:` \
/
_output_shapes
:���������
)
_user_specified_nameteacher_inputs_
�
�
K__inference_teacher_outputs_layer_call_and_return_conditional_losses_116316

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�

�
L__inference_teacher_conv2d_3_layer_call_and_return_conditional_losses_115750

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�L
�
I__inference_teacher_scn_2_layer_call_and_return_conditional_losses_115850
teacher_inputs_1
teacher_conv2d_1_115805:%
teacher_conv2d_1_115807:1
teacher_conv2d_2_115812:%
teacher_conv2d_2_115814:(
teacher_latent_115819:	�P#
teacher_latent_115821:P'
teacher_dense_115824:	P�#
teacher_dense_115826:	�1
teacher_conv2d_3_115831:%
teacher_conv2d_3_115833:1
teacher_conv2d_4_115838:%
teacher_conv2d_4_115840:0
teacher_outputs_115844:$
teacher_outputs_115846:
identity��(teacher_conv2d_1/StatefulPartitionedCall�(teacher_conv2d_2/StatefulPartitionedCall�(teacher_conv2d_3/StatefulPartitionedCall�(teacher_conv2d_4/StatefulPartitionedCall�%teacher_dense/StatefulPartitionedCall�&teacher_latent/StatefulPartitionedCall�'teacher_outputs/StatefulPartitionedCall�
teacher_reshape/PartitionedCallPartitionedCallteacher_inputs_*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_teacher_reshape_layer_call_and_return_conditional_losses_115637�
(teacher_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(teacher_reshape/PartitionedCall:output:0teacher_conv2d_1_115805teacher_conv2d_1_115807*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_teacher_conv2d_1_layer_call_and_return_conditional_losses_115648�
teacher_relu_1/PartitionedCallPartitionedCall1teacher_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_relu_1_layer_call_and_return_conditional_losses_115658�
teacher_pool_1/PartitionedCallPartitionedCall'teacher_relu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_pool_1_layer_call_and_return_conditional_losses_115599�
(teacher_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall'teacher_pool_1/PartitionedCall:output:0teacher_conv2d_2_115812teacher_conv2d_2_115814*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_teacher_conv2d_2_layer_call_and_return_conditional_losses_115670�
teacher_relu_2/PartitionedCallPartitionedCall1teacher_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_relu_2_layer_call_and_return_conditional_losses_115680�
teacher_flatten/PartitionedCallPartitionedCall'teacher_relu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_teacher_flatten_layer_call_and_return_conditional_losses_115687�
&teacher_latent/StatefulPartitionedCallStatefulPartitionedCall(teacher_flatten/PartitionedCall:output:0teacher_latent_115819teacher_latent_115821*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_latent_layer_call_and_return_conditional_losses_115699�
%teacher_dense/StatefulPartitionedCallStatefulPartitionedCall/teacher_latent/StatefulPartitionedCall:output:0teacher_dense_115824teacher_dense_115826*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_teacher_dense_layer_call_and_return_conditional_losses_115714�
 teacher_reshape2/PartitionedCallPartitionedCall.teacher_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_teacher_reshape2_layer_call_and_return_conditional_losses_115733�
teacher_relu_3/PartitionedCallPartitionedCall)teacher_reshape2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_relu_3_layer_call_and_return_conditional_losses_115739�
(teacher_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall'teacher_relu_3/PartitionedCall:output:0teacher_conv2d_3_115831teacher_conv2d_3_115833*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_teacher_conv2d_3_layer_call_and_return_conditional_losses_115750�
teacher_relu_4/PartitionedCallPartitionedCall1teacher_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_relu_4_layer_call_and_return_conditional_losses_115760�
"teacher_upsampling/PartitionedCallPartitionedCall'teacher_relu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_teacher_upsampling_layer_call_and_return_conditional_losses_115616�
(teacher_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall+teacher_upsampling/PartitionedCall:output:0teacher_conv2d_4_115838teacher_conv2d_4_115840*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_teacher_conv2d_4_layer_call_and_return_conditional_losses_115772�
teacher_relu_5/PartitionedCallPartitionedCall1teacher_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_relu_5_layer_call_and_return_conditional_losses_115782�
'teacher_outputs/StatefulPartitionedCallStatefulPartitionedCall'teacher_relu_5/PartitionedCall:output:0teacher_outputs_115844teacher_outputs_115846*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_teacher_outputs_layer_call_and_return_conditional_losses_115794�
IdentityIdentity0teacher_outputs/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp)^teacher_conv2d_1/StatefulPartitionedCall)^teacher_conv2d_2/StatefulPartitionedCall)^teacher_conv2d_3/StatefulPartitionedCall)^teacher_conv2d_4/StatefulPartitionedCall&^teacher_dense/StatefulPartitionedCall'^teacher_latent/StatefulPartitionedCall(^teacher_outputs/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : 2T
(teacher_conv2d_1/StatefulPartitionedCall(teacher_conv2d_1/StatefulPartitionedCall2T
(teacher_conv2d_2/StatefulPartitionedCall(teacher_conv2d_2/StatefulPartitionedCall2T
(teacher_conv2d_3/StatefulPartitionedCall(teacher_conv2d_3/StatefulPartitionedCall2T
(teacher_conv2d_4/StatefulPartitionedCall(teacher_conv2d_4/StatefulPartitionedCall2N
%teacher_dense/StatefulPartitionedCall%teacher_dense/StatefulPartitionedCall2P
&teacher_latent/StatefulPartitionedCall&teacher_latent/StatefulPartitionedCall2R
'teacher_outputs/StatefulPartitionedCall'teacher_outputs/StatefulPartitionedCall:&"
 
_user_specified_name115846:&"
 
_user_specified_name115844:&"
 
_user_specified_name115840:&"
 
_user_specified_name115838:&
"
 
_user_specified_name115833:&	"
 
_user_specified_name115831:&"
 
_user_specified_name115826:&"
 
_user_specified_name115824:&"
 
_user_specified_name115821:&"
 
_user_specified_name115819:&"
 
_user_specified_name115814:&"
 
_user_specified_name115812:&"
 
_user_specified_name115807:&"
 
_user_specified_name115805:` \
/
_output_shapes
:���������
)
_user_specified_nameteacher_inputs_
�
�
1__inference_teacher_conv2d_1_layer_call_fn_116083

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_teacher_conv2d_1_layer_call_and_return_conditional_losses_115648w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name116079:&"
 
_user_specified_name116077:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
J__inference_teacher_relu_5_layer_call_and_return_conditional_losses_116296

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+���������������������������t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
h
L__inference_teacher_reshape2_layer_call_and_return_conditional_losses_116211

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
K__inference_teacher_outputs_layer_call_and_return_conditional_losses_115794

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
0__inference_teacher_outputs_layer_call_fn_116305

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_teacher_outputs_layer_call_and_return_conditional_losses_115794�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name116301:&"
 
_user_specified_name116299:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�

�
J__inference_teacher_latent_layer_call_and_return_conditional_losses_115699

inputs1
matmul_readvariableop_resource:	�P-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������PP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������Pa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������PS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
g
K__inference_teacher_flatten_layer_call_and_return_conditional_losses_116153

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����v  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
J__inference_teacher_relu_2_layer_call_and_return_conditional_losses_116142

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
g
K__inference_teacher_reshape_layer_call_and_return_conditional_losses_116074

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
K
/__inference_teacher_relu_1_layer_call_fn_116098

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_relu_1_layer_call_and_return_conditional_losses_115658h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
J__inference_teacher_relu_1_layer_call_and_return_conditional_losses_116103

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_116761
file_prefixB
(assignvariableop_teacher_conv2d_1_kernel:6
(assignvariableop_1_teacher_conv2d_1_bias:D
*assignvariableop_2_teacher_conv2d_2_kernel:6
(assignvariableop_3_teacher_conv2d_2_bias:;
(assignvariableop_4_teacher_latent_kernel:	�P4
&assignvariableop_5_teacher_latent_bias:P:
'assignvariableop_6_teacher_dense_kernel:	P�4
%assignvariableop_7_teacher_dense_bias:	�D
*assignvariableop_8_teacher_conv2d_3_kernel:6
(assignvariableop_9_teacher_conv2d_3_bias:E
+assignvariableop_10_teacher_conv2d_4_kernel:7
)assignvariableop_11_teacher_conv2d_4_bias:D
*assignvariableop_12_teacher_outputs_kernel:6
(assignvariableop_13_teacher_outputs_bias:'
assignvariableop_14_iteration:	 +
!assignvariableop_15_learning_rate: L
2assignvariableop_16_adam_m_teacher_conv2d_1_kernel:L
2assignvariableop_17_adam_v_teacher_conv2d_1_kernel:>
0assignvariableop_18_adam_m_teacher_conv2d_1_bias:>
0assignvariableop_19_adam_v_teacher_conv2d_1_bias:L
2assignvariableop_20_adam_m_teacher_conv2d_2_kernel:L
2assignvariableop_21_adam_v_teacher_conv2d_2_kernel:>
0assignvariableop_22_adam_m_teacher_conv2d_2_bias:>
0assignvariableop_23_adam_v_teacher_conv2d_2_bias:C
0assignvariableop_24_adam_m_teacher_latent_kernel:	�PC
0assignvariableop_25_adam_v_teacher_latent_kernel:	�P<
.assignvariableop_26_adam_m_teacher_latent_bias:P<
.assignvariableop_27_adam_v_teacher_latent_bias:PB
/assignvariableop_28_adam_m_teacher_dense_kernel:	P�B
/assignvariableop_29_adam_v_teacher_dense_kernel:	P�<
-assignvariableop_30_adam_m_teacher_dense_bias:	�<
-assignvariableop_31_adam_v_teacher_dense_bias:	�L
2assignvariableop_32_adam_m_teacher_conv2d_3_kernel:L
2assignvariableop_33_adam_v_teacher_conv2d_3_kernel:>
0assignvariableop_34_adam_m_teacher_conv2d_3_bias:>
0assignvariableop_35_adam_v_teacher_conv2d_3_bias:L
2assignvariableop_36_adam_m_teacher_conv2d_4_kernel:L
2assignvariableop_37_adam_v_teacher_conv2d_4_kernel:>
0assignvariableop_38_adam_m_teacher_conv2d_4_bias:>
0assignvariableop_39_adam_v_teacher_conv2d_4_bias:K
1assignvariableop_40_adam_m_teacher_outputs_kernel:K
1assignvariableop_41_adam_v_teacher_outputs_kernel:=
/assignvariableop_42_adam_m_teacher_outputs_bias:=
/assignvariableop_43_adam_v_teacher_outputs_bias:#
assignvariableop_44_total: #
assignvariableop_45_count: 
identity_47��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*�
value�B�/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::*=
dtypes3
12/	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp(assignvariableop_teacher_conv2d_1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp(assignvariableop_1_teacher_conv2d_1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp*assignvariableop_2_teacher_conv2d_2_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp(assignvariableop_3_teacher_conv2d_2_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp(assignvariableop_4_teacher_latent_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp&assignvariableop_5_teacher_latent_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp'assignvariableop_6_teacher_dense_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp%assignvariableop_7_teacher_dense_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp*assignvariableop_8_teacher_conv2d_3_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp(assignvariableop_9_teacher_conv2d_3_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp+assignvariableop_10_teacher_conv2d_4_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp)assignvariableop_11_teacher_conv2d_4_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp*assignvariableop_12_teacher_outputs_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp(assignvariableop_13_teacher_outputs_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_iterationIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_learning_rateIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp2assignvariableop_16_adam_m_teacher_conv2d_1_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp2assignvariableop_17_adam_v_teacher_conv2d_1_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp0assignvariableop_18_adam_m_teacher_conv2d_1_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp0assignvariableop_19_adam_v_teacher_conv2d_1_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp2assignvariableop_20_adam_m_teacher_conv2d_2_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_v_teacher_conv2d_2_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp0assignvariableop_22_adam_m_teacher_conv2d_2_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp0assignvariableop_23_adam_v_teacher_conv2d_2_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp0assignvariableop_24_adam_m_teacher_latent_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp0assignvariableop_25_adam_v_teacher_latent_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp.assignvariableop_26_adam_m_teacher_latent_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp.assignvariableop_27_adam_v_teacher_latent_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp/assignvariableop_28_adam_m_teacher_dense_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp/assignvariableop_29_adam_v_teacher_dense_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp-assignvariableop_30_adam_m_teacher_dense_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_v_teacher_dense_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp2assignvariableop_32_adam_m_teacher_conv2d_3_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp2assignvariableop_33_adam_v_teacher_conv2d_3_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp0assignvariableop_34_adam_m_teacher_conv2d_3_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp0assignvariableop_35_adam_v_teacher_conv2d_3_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp2assignvariableop_36_adam_m_teacher_conv2d_4_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp2assignvariableop_37_adam_v_teacher_conv2d_4_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp0assignvariableop_38_adam_m_teacher_conv2d_4_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp0assignvariableop_39_adam_v_teacher_conv2d_4_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp1assignvariableop_40_adam_m_teacher_outputs_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp1assignvariableop_41_adam_v_teacher_outputs_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp/assignvariableop_42_adam_m_teacher_outputs_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp/assignvariableop_43_adam_v_teacher_outputs_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOpassignvariableop_44_totalIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOpassignvariableop_45_countIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_46Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_47IdentityIdentity_46:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_47Identity_47:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%.!

_user_specified_namecount:%-!

_user_specified_nametotal:;,7
5
_user_specified_nameAdam/v/teacher_outputs/bias:;+7
5
_user_specified_nameAdam/m/teacher_outputs/bias:=*9
7
_user_specified_nameAdam/v/teacher_outputs/kernel:=)9
7
_user_specified_nameAdam/m/teacher_outputs/kernel:<(8
6
_user_specified_nameAdam/v/teacher_conv2d_4/bias:<'8
6
_user_specified_nameAdam/m/teacher_conv2d_4/bias:>&:
8
_user_specified_name Adam/v/teacher_conv2d_4/kernel:>%:
8
_user_specified_name Adam/m/teacher_conv2d_4/kernel:<$8
6
_user_specified_nameAdam/v/teacher_conv2d_3/bias:<#8
6
_user_specified_nameAdam/m/teacher_conv2d_3/bias:>":
8
_user_specified_name Adam/v/teacher_conv2d_3/kernel:>!:
8
_user_specified_name Adam/m/teacher_conv2d_3/kernel:9 5
3
_user_specified_nameAdam/v/teacher_dense/bias:95
3
_user_specified_nameAdam/m/teacher_dense/bias:;7
5
_user_specified_nameAdam/v/teacher_dense/kernel:;7
5
_user_specified_nameAdam/m/teacher_dense/kernel::6
4
_user_specified_nameAdam/v/teacher_latent/bias::6
4
_user_specified_nameAdam/m/teacher_latent/bias:<8
6
_user_specified_nameAdam/v/teacher_latent/kernel:<8
6
_user_specified_nameAdam/m/teacher_latent/kernel:<8
6
_user_specified_nameAdam/v/teacher_conv2d_2/bias:<8
6
_user_specified_nameAdam/m/teacher_conv2d_2/bias:>:
8
_user_specified_name Adam/v/teacher_conv2d_2/kernel:>:
8
_user_specified_name Adam/m/teacher_conv2d_2/kernel:<8
6
_user_specified_nameAdam/v/teacher_conv2d_1/bias:<8
6
_user_specified_nameAdam/m/teacher_conv2d_1/bias:>:
8
_user_specified_name Adam/v/teacher_conv2d_1/kernel:>:
8
_user_specified_name Adam/m/teacher_conv2d_1/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:40
.
_user_specified_nameteacher_outputs/bias:62
0
_user_specified_nameteacher_outputs/kernel:51
/
_user_specified_nameteacher_conv2d_4/bias:73
1
_user_specified_nameteacher_conv2d_4/kernel:5
1
/
_user_specified_nameteacher_conv2d_3/bias:7	3
1
_user_specified_nameteacher_conv2d_3/kernel:2.
,
_user_specified_nameteacher_dense/bias:40
.
_user_specified_nameteacher_dense/kernel:3/
-
_user_specified_nameteacher_latent/bias:51
/
_user_specified_nameteacher_latent/kernel:51
/
_user_specified_nameteacher_conv2d_2/bias:73
1
_user_specified_nameteacher_conv2d_2/kernel:51
/
_user_specified_nameteacher_conv2d_1/bias:73
1
_user_specified_nameteacher_conv2d_1/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
j
N__inference_teacher_upsampling_layer_call_and_return_conditional_losses_116267

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
f
J__inference_teacher_relu_1_layer_call_and_return_conditional_losses_115658

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
J__inference_teacher_relu_2_layer_call_and_return_conditional_losses_115680

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
1__inference_teacher_conv2d_2_layer_call_fn_116122

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_teacher_conv2d_2_layer_call_and_return_conditional_losses_115670w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name116118:&"
 
_user_specified_name116116:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_teacher_scn_2_layer_call_fn_115883
teacher_inputs_!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	�P
	unknown_4:P
	unknown_5:	P�
	unknown_6:	�#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallteacher_inputs_unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_teacher_scn_2_layer_call_and_return_conditional_losses_115801�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name115879:&"
 
_user_specified_name115877:&"
 
_user_specified_name115875:&"
 
_user_specified_name115873:&
"
 
_user_specified_name115871:&	"
 
_user_specified_name115869:&"
 
_user_specified_name115867:&"
 
_user_specified_name115865:&"
 
_user_specified_name115863:&"
 
_user_specified_name115861:&"
 
_user_specified_name115859:&"
 
_user_specified_name115857:&"
 
_user_specified_name115855:&"
 
_user_specified_name115853:` \
/
_output_shapes
:���������
)
_user_specified_nameteacher_inputs_
�
L
0__inference_teacher_flatten_layer_call_fn_116147

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
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_teacher_flatten_layer_call_and_return_conditional_losses_115687a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�L
�
I__inference_teacher_scn_2_layer_call_and_return_conditional_losses_115801
teacher_inputs_1
teacher_conv2d_1_115649:%
teacher_conv2d_1_115651:1
teacher_conv2d_2_115671:%
teacher_conv2d_2_115673:(
teacher_latent_115700:	�P#
teacher_latent_115702:P'
teacher_dense_115715:	P�#
teacher_dense_115717:	�1
teacher_conv2d_3_115751:%
teacher_conv2d_3_115753:1
teacher_conv2d_4_115773:%
teacher_conv2d_4_115775:0
teacher_outputs_115795:$
teacher_outputs_115797:
identity��(teacher_conv2d_1/StatefulPartitionedCall�(teacher_conv2d_2/StatefulPartitionedCall�(teacher_conv2d_3/StatefulPartitionedCall�(teacher_conv2d_4/StatefulPartitionedCall�%teacher_dense/StatefulPartitionedCall�&teacher_latent/StatefulPartitionedCall�'teacher_outputs/StatefulPartitionedCall�
teacher_reshape/PartitionedCallPartitionedCallteacher_inputs_*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_teacher_reshape_layer_call_and_return_conditional_losses_115637�
(teacher_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(teacher_reshape/PartitionedCall:output:0teacher_conv2d_1_115649teacher_conv2d_1_115651*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_teacher_conv2d_1_layer_call_and_return_conditional_losses_115648�
teacher_relu_1/PartitionedCallPartitionedCall1teacher_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_relu_1_layer_call_and_return_conditional_losses_115658�
teacher_pool_1/PartitionedCallPartitionedCall'teacher_relu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_pool_1_layer_call_and_return_conditional_losses_115599�
(teacher_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall'teacher_pool_1/PartitionedCall:output:0teacher_conv2d_2_115671teacher_conv2d_2_115673*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_teacher_conv2d_2_layer_call_and_return_conditional_losses_115670�
teacher_relu_2/PartitionedCallPartitionedCall1teacher_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_relu_2_layer_call_and_return_conditional_losses_115680�
teacher_flatten/PartitionedCallPartitionedCall'teacher_relu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_teacher_flatten_layer_call_and_return_conditional_losses_115687�
&teacher_latent/StatefulPartitionedCallStatefulPartitionedCall(teacher_flatten/PartitionedCall:output:0teacher_latent_115700teacher_latent_115702*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_latent_layer_call_and_return_conditional_losses_115699�
%teacher_dense/StatefulPartitionedCallStatefulPartitionedCall/teacher_latent/StatefulPartitionedCall:output:0teacher_dense_115715teacher_dense_115717*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_teacher_dense_layer_call_and_return_conditional_losses_115714�
 teacher_reshape2/PartitionedCallPartitionedCall.teacher_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_teacher_reshape2_layer_call_and_return_conditional_losses_115733�
teacher_relu_3/PartitionedCallPartitionedCall)teacher_reshape2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_relu_3_layer_call_and_return_conditional_losses_115739�
(teacher_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall'teacher_relu_3/PartitionedCall:output:0teacher_conv2d_3_115751teacher_conv2d_3_115753*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_teacher_conv2d_3_layer_call_and_return_conditional_losses_115750�
teacher_relu_4/PartitionedCallPartitionedCall1teacher_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_relu_4_layer_call_and_return_conditional_losses_115760�
"teacher_upsampling/PartitionedCallPartitionedCall'teacher_relu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_teacher_upsampling_layer_call_and_return_conditional_losses_115616�
(teacher_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall+teacher_upsampling/PartitionedCall:output:0teacher_conv2d_4_115773teacher_conv2d_4_115775*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_teacher_conv2d_4_layer_call_and_return_conditional_losses_115772�
teacher_relu_5/PartitionedCallPartitionedCall1teacher_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_relu_5_layer_call_and_return_conditional_losses_115782�
'teacher_outputs/StatefulPartitionedCallStatefulPartitionedCall'teacher_relu_5/PartitionedCall:output:0teacher_outputs_115795teacher_outputs_115797*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_teacher_outputs_layer_call_and_return_conditional_losses_115794�
IdentityIdentity0teacher_outputs/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp)^teacher_conv2d_1/StatefulPartitionedCall)^teacher_conv2d_2/StatefulPartitionedCall)^teacher_conv2d_3/StatefulPartitionedCall)^teacher_conv2d_4/StatefulPartitionedCall&^teacher_dense/StatefulPartitionedCall'^teacher_latent/StatefulPartitionedCall(^teacher_outputs/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : 2T
(teacher_conv2d_1/StatefulPartitionedCall(teacher_conv2d_1/StatefulPartitionedCall2T
(teacher_conv2d_2/StatefulPartitionedCall(teacher_conv2d_2/StatefulPartitionedCall2T
(teacher_conv2d_3/StatefulPartitionedCall(teacher_conv2d_3/StatefulPartitionedCall2T
(teacher_conv2d_4/StatefulPartitionedCall(teacher_conv2d_4/StatefulPartitionedCall2N
%teacher_dense/StatefulPartitionedCall%teacher_dense/StatefulPartitionedCall2P
&teacher_latent/StatefulPartitionedCall&teacher_latent/StatefulPartitionedCall2R
'teacher_outputs/StatefulPartitionedCall'teacher_outputs/StatefulPartitionedCall:&"
 
_user_specified_name115797:&"
 
_user_specified_name115795:&"
 
_user_specified_name115775:&"
 
_user_specified_name115773:&
"
 
_user_specified_name115753:&	"
 
_user_specified_name115751:&"
 
_user_specified_name115717:&"
 
_user_specified_name115715:&"
 
_user_specified_name115702:&"
 
_user_specified_name115700:&"
 
_user_specified_name115673:&"
 
_user_specified_name115671:&"
 
_user_specified_name115651:&"
 
_user_specified_name115649:` \
/
_output_shapes
:���������
)
_user_specified_nameteacher_inputs_
�
f
J__inference_teacher_relu_3_layer_call_and_return_conditional_losses_116221

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
L
0__inference_teacher_reshape_layer_call_fn_116060

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_teacher_reshape_layer_call_and_return_conditional_losses_115637h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_teacher_scn_2_layer_call_fn_115916
teacher_inputs_!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	�P
	unknown_4:P
	unknown_5:	P�
	unknown_6:	�#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallteacher_inputs_unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_teacher_scn_2_layer_call_and_return_conditional_losses_115850�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name115912:&"
 
_user_specified_name115910:&"
 
_user_specified_name115908:&"
 
_user_specified_name115906:&
"
 
_user_specified_name115904:&	"
 
_user_specified_name115902:&"
 
_user_specified_name115900:&"
 
_user_specified_name115898:&"
 
_user_specified_name115896:&"
 
_user_specified_name115894:&"
 
_user_specified_name115892:&"
 
_user_specified_name115890:&"
 
_user_specified_name115888:&"
 
_user_specified_name115886:` \
/
_output_shapes
:���������
)
_user_specified_nameteacher_inputs_
�
K
/__inference_teacher_relu_5_layer_call_fn_116291

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_relu_5_layer_call_and_return_conditional_losses_115782z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+���������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+���������������������������:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
/__inference_teacher_latent_layer_call_fn_116162

inputs
unknown:	�P
	unknown_0:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_teacher_latent_layer_call_and_return_conditional_losses_115699o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name116158:&"
 
_user_specified_name116156:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
S
teacher_inputs_@
!serving_default_teacher_inputs_:0���������K
teacher_outputs8
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer_with_weights-4
layer-12
layer-13
layer-14
layer_with_weights-5
layer-15
layer-16
layer_with_weights-6
layer-17
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias
 *_jit_compiled_convolution_op"
_tf_keras_layer
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias
 ?_jit_compiled_convolution_op"
_tf_keras_layer
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer
�
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses

Rkernel
Sbias"
_tf_keras_layer
�
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

Zkernel
[bias"
_tf_keras_layer
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_layer
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses

nkernel
obias
 p_jit_compiled_convolution_op"
_tf_keras_layer
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layer
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
(0
)1
=2
>3
R4
S5
Z6
[7
n8
o9
�10
�11
�12
�13"
trackable_list_wrapper
�
(0
)1
=2
>3
R4
S5
Z6
[7
n8
o9
�10
�11
�12
�13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
.__inference_teacher_scn_2_layer_call_fn_115883
.__inference_teacher_scn_2_layer_call_fn_115916�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
I__inference_teacher_scn_2_layer_call_and_return_conditional_losses_115801
I__inference_teacher_scn_2_layer_call_and_return_conditional_losses_115850�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�B�
!__inference__wrapped_model_115594teacher_inputs_"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_teacher_reshape_layer_call_fn_116060�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
K__inference_teacher_reshape_layer_call_and_return_conditional_losses_116074�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_teacher_conv2d_1_layer_call_fn_116083�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
L__inference_teacher_conv2d_1_layer_call_and_return_conditional_losses_116093�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
1:/2teacher_conv2d_1/kernel
#:!2teacher_conv2d_1/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_teacher_relu_1_layer_call_fn_116098�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
J__inference_teacher_relu_1_layer_call_and_return_conditional_losses_116103�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_teacher_pool_1_layer_call_fn_116108�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
J__inference_teacher_pool_1_layer_call_and_return_conditional_losses_116113�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_teacher_conv2d_2_layer_call_fn_116122�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
L__inference_teacher_conv2d_2_layer_call_and_return_conditional_losses_116132�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
1:/2teacher_conv2d_2/kernel
#:!2teacher_conv2d_2/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_teacher_relu_2_layer_call_fn_116137�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
J__inference_teacher_relu_2_layer_call_and_return_conditional_losses_116142�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_teacher_flatten_layer_call_fn_116147�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
K__inference_teacher_flatten_layer_call_and_return_conditional_losses_116153�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_teacher_latent_layer_call_fn_116162�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
J__inference_teacher_latent_layer_call_and_return_conditional_losses_116173�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
(:&	�P2teacher_latent/kernel
!:P2teacher_latent/bias
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_teacher_dense_layer_call_fn_116182�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
I__inference_teacher_dense_layer_call_and_return_conditional_losses_116192�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
':%	P�2teacher_dense/kernel
!:�2teacher_dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_teacher_reshape2_layer_call_fn_116197�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
L__inference_teacher_reshape2_layer_call_and_return_conditional_losses_116211�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_teacher_relu_3_layer_call_fn_116216�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
J__inference_teacher_relu_3_layer_call_and_return_conditional_losses_116221�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_teacher_conv2d_3_layer_call_fn_116230�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
L__inference_teacher_conv2d_3_layer_call_and_return_conditional_losses_116240�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
1:/2teacher_conv2d_3/kernel
#:!2teacher_conv2d_3/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_teacher_relu_4_layer_call_fn_116245�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
J__inference_teacher_relu_4_layer_call_and_return_conditional_losses_116250�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_teacher_upsampling_layer_call_fn_116255�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
N__inference_teacher_upsampling_layer_call_and_return_conditional_losses_116267�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_teacher_conv2d_4_layer_call_fn_116276�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
L__inference_teacher_conv2d_4_layer_call_and_return_conditional_losses_116286�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
1:/2teacher_conv2d_4/kernel
#:!2teacher_conv2d_4/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_teacher_relu_5_layer_call_fn_116291�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
J__inference_teacher_relu_5_layer_call_and_return_conditional_losses_116296�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_teacher_outputs_layer_call_fn_116305�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
K__inference_teacher_outputs_layer_call_and_return_conditional_losses_116316�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
0:.2teacher_outputs/kernel
": 2teacher_outputs/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_teacher_scn_2_layer_call_fn_115883teacher_inputs_"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_teacher_scn_2_layer_call_fn_115916teacher_inputs_"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_teacher_scn_2_layer_call_and_return_conditional_losses_115801teacher_inputs_"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_teacher_scn_2_layer_call_and_return_conditional_losses_115850teacher_inputs_"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_116055teacher_inputs_"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 $

kwonlyargs�
jteacher_inputs_
kwonlydefaults
 
annotations� *
 
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
�B�
0__inference_teacher_reshape_layer_call_fn_116060inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
K__inference_teacher_reshape_layer_call_and_return_conditional_losses_116074inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
1__inference_teacher_conv2d_1_layer_call_fn_116083inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
L__inference_teacher_conv2d_1_layer_call_and_return_conditional_losses_116093inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
/__inference_teacher_relu_1_layer_call_fn_116098inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
J__inference_teacher_relu_1_layer_call_and_return_conditional_losses_116103inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
/__inference_teacher_pool_1_layer_call_fn_116108inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
J__inference_teacher_pool_1_layer_call_and_return_conditional_losses_116113inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
1__inference_teacher_conv2d_2_layer_call_fn_116122inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
L__inference_teacher_conv2d_2_layer_call_and_return_conditional_losses_116132inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
/__inference_teacher_relu_2_layer_call_fn_116137inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
J__inference_teacher_relu_2_layer_call_and_return_conditional_losses_116142inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
0__inference_teacher_flatten_layer_call_fn_116147inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
K__inference_teacher_flatten_layer_call_and_return_conditional_losses_116153inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
/__inference_teacher_latent_layer_call_fn_116162inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
J__inference_teacher_latent_layer_call_and_return_conditional_losses_116173inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
.__inference_teacher_dense_layer_call_fn_116182inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
I__inference_teacher_dense_layer_call_and_return_conditional_losses_116192inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
1__inference_teacher_reshape2_layer_call_fn_116197inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
L__inference_teacher_reshape2_layer_call_and_return_conditional_losses_116211inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
/__inference_teacher_relu_3_layer_call_fn_116216inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
J__inference_teacher_relu_3_layer_call_and_return_conditional_losses_116221inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
1__inference_teacher_conv2d_3_layer_call_fn_116230inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
L__inference_teacher_conv2d_3_layer_call_and_return_conditional_losses_116240inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
/__inference_teacher_relu_4_layer_call_fn_116245inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
J__inference_teacher_relu_4_layer_call_and_return_conditional_losses_116250inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
3__inference_teacher_upsampling_layer_call_fn_116255inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
N__inference_teacher_upsampling_layer_call_and_return_conditional_losses_116267inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
1__inference_teacher_conv2d_4_layer_call_fn_116276inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
L__inference_teacher_conv2d_4_layer_call_and_return_conditional_losses_116286inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
/__inference_teacher_relu_5_layer_call_fn_116291inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
J__inference_teacher_relu_5_layer_call_and_return_conditional_losses_116296inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
0__inference_teacher_outputs_layer_call_fn_116305inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
K__inference_teacher_outputs_layer_call_and_return_conditional_losses_116316inputs"�
���
FullArgSpec
args�

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
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
6:42Adam/m/teacher_conv2d_1/kernel
6:42Adam/v/teacher_conv2d_1/kernel
(:&2Adam/m/teacher_conv2d_1/bias
(:&2Adam/v/teacher_conv2d_1/bias
6:42Adam/m/teacher_conv2d_2/kernel
6:42Adam/v/teacher_conv2d_2/kernel
(:&2Adam/m/teacher_conv2d_2/bias
(:&2Adam/v/teacher_conv2d_2/bias
-:+	�P2Adam/m/teacher_latent/kernel
-:+	�P2Adam/v/teacher_latent/kernel
&:$P2Adam/m/teacher_latent/bias
&:$P2Adam/v/teacher_latent/bias
,:*	P�2Adam/m/teacher_dense/kernel
,:*	P�2Adam/v/teacher_dense/kernel
&:$�2Adam/m/teacher_dense/bias
&:$�2Adam/v/teacher_dense/bias
6:42Adam/m/teacher_conv2d_3/kernel
6:42Adam/v/teacher_conv2d_3/kernel
(:&2Adam/m/teacher_conv2d_3/bias
(:&2Adam/v/teacher_conv2d_3/bias
6:42Adam/m/teacher_conv2d_4/kernel
6:42Adam/v/teacher_conv2d_4/kernel
(:&2Adam/m/teacher_conv2d_4/bias
(:&2Adam/v/teacher_conv2d_4/bias
5:32Adam/m/teacher_outputs/kernel
5:32Adam/v/teacher_outputs/kernel
':%2Adam/m/teacher_outputs/bias
':%2Adam/v/teacher_outputs/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count�
!__inference__wrapped_model_115594�()=>RSZ[no����@�=
6�3
1�.
teacher_inputs_���������
� "I�F
D
teacher_outputs1�.
teacher_outputs����������
$__inference_signature_wrapper_116055�()=>RSZ[no����S�P
� 
I�F
D
teacher_inputs_1�.
teacher_inputs_���������"I�F
D
teacher_outputs1�.
teacher_outputs����������
L__inference_teacher_conv2d_1_layer_call_and_return_conditional_losses_116093s()7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
1__inference_teacher_conv2d_1_layer_call_fn_116083h()7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
L__inference_teacher_conv2d_2_layer_call_and_return_conditional_losses_116132s=>7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
1__inference_teacher_conv2d_2_layer_call_fn_116122h=>7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
L__inference_teacher_conv2d_3_layer_call_and_return_conditional_losses_116240sno7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
1__inference_teacher_conv2d_3_layer_call_fn_116230hno7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
L__inference_teacher_conv2d_4_layer_call_and_return_conditional_losses_116286���I�F
?�<
:�7
inputs+���������������������������
� "F�C
<�9
tensor_0+���������������������������
� �
1__inference_teacher_conv2d_4_layer_call_fn_116276���I�F
?�<
:�7
inputs+���������������������������
� ";�8
unknown+����������������������������
I__inference_teacher_dense_layer_call_and_return_conditional_losses_116192dZ[/�,
%�"
 �
inputs���������P
� "-�*
#� 
tensor_0����������
� �
.__inference_teacher_dense_layer_call_fn_116182YZ[/�,
%�"
 �
inputs���������P
� ""�
unknown�����������
K__inference_teacher_flatten_layer_call_and_return_conditional_losses_116153h7�4
-�*
(�%
inputs���������
� "-�*
#� 
tensor_0����������
� �
0__inference_teacher_flatten_layer_call_fn_116147]7�4
-�*
(�%
inputs���������
� ""�
unknown�����������
J__inference_teacher_latent_layer_call_and_return_conditional_losses_116173dRS0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������P
� �
/__inference_teacher_latent_layer_call_fn_116162YRS0�-
&�#
!�
inputs����������
� "!�
unknown���������P�
K__inference_teacher_outputs_layer_call_and_return_conditional_losses_116316���I�F
?�<
:�7
inputs+���������������������������
� "F�C
<�9
tensor_0+���������������������������
� �
0__inference_teacher_outputs_layer_call_fn_116305���I�F
?�<
:�7
inputs+���������������������������
� ";�8
unknown+����������������������������
J__inference_teacher_pool_1_layer_call_and_return_conditional_losses_116113�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
/__inference_teacher_pool_1_layer_call_fn_116108�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
J__inference_teacher_relu_1_layer_call_and_return_conditional_losses_116103o7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
/__inference_teacher_relu_1_layer_call_fn_116098d7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
J__inference_teacher_relu_2_layer_call_and_return_conditional_losses_116142o7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
/__inference_teacher_relu_2_layer_call_fn_116137d7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
J__inference_teacher_relu_3_layer_call_and_return_conditional_losses_116221o7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
/__inference_teacher_relu_3_layer_call_fn_116216d7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
J__inference_teacher_relu_4_layer_call_and_return_conditional_losses_116250o7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
/__inference_teacher_relu_4_layer_call_fn_116245d7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
J__inference_teacher_relu_5_layer_call_and_return_conditional_losses_116296�I�F
?�<
:�7
inputs+���������������������������
� "F�C
<�9
tensor_0+���������������������������
� �
/__inference_teacher_relu_5_layer_call_fn_116291�I�F
?�<
:�7
inputs+���������������������������
� ";�8
unknown+����������������������������
L__inference_teacher_reshape2_layer_call_and_return_conditional_losses_116211h0�-
&�#
!�
inputs����������
� "4�1
*�'
tensor_0���������
� �
1__inference_teacher_reshape2_layer_call_fn_116197]0�-
&�#
!�
inputs����������
� ")�&
unknown����������
K__inference_teacher_reshape_layer_call_and_return_conditional_losses_116074o7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
0__inference_teacher_reshape_layer_call_fn_116060d7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
I__inference_teacher_scn_2_layer_call_and_return_conditional_losses_115801�()=>RSZ[no����H�E
>�;
1�.
teacher_inputs_���������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
I__inference_teacher_scn_2_layer_call_and_return_conditional_losses_115850�()=>RSZ[no����H�E
>�;
1�.
teacher_inputs_���������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
.__inference_teacher_scn_2_layer_call_fn_115883�()=>RSZ[no����H�E
>�;
1�.
teacher_inputs_���������
p

 
� ";�8
unknown+����������������������������
.__inference_teacher_scn_2_layer_call_fn_115916�()=>RSZ[no����H�E
>�;
1�.
teacher_inputs_���������
p 

 
� ";�8
unknown+����������������������������
N__inference_teacher_upsampling_layer_call_and_return_conditional_losses_116267�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
3__inference_teacher_upsampling_layer_call_fn_116255�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4������������������������������������