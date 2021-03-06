?	\W?o??@\W?o??@!\W?o??@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC\W?o??@#h?$?U@1?릔?W?@AĲ?CR??I?.l?V?	@rEagerKernelExecute 0*	>
ף?A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator????~d@!O?Ն??X@)????~d@1O?Ն??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism-%?I(}??!;THv??)cd?˻??1Z??~L??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???v?>??!(?o??)???v?>??1(?o??:Preprocessing2F
Iterator::Model?Tl?눫?!??Y??ɠ?)?|	^p?1?5??7?c?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap(??9?~d@!?,???X@)?{G?	1g?1?r9VG\?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI@??I?a??Q?ْy?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	#h?$?U@#h?$?U@!#h?$?U@      ??!       "	?릔?W?@?릔?W?@!?릔?W?@*      ??!       2	Ĳ?CR??Ĳ?CR??!Ĳ?CR??:	?.l?V?	@?.l?V?	@!?.l?V?	@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@??I?a??y?ْy?X@?"k
?gradient_tape/model_1/res5a_branch1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?6??Ov??!?6??Ov??0"k
?gradient_tape/model_1/res4a_branch1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???)K???!?w?u̝?0"l
@gradient_tape/model_1/res5c_branch2b/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterb?;????!4!??d???0"l
@gradient_tape/model_1/res5b_branch2b/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterDc8????!7??Y?=??0"l
@gradient_tape/model_1/res5a_branch2b/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterrEvz??!?C5w*ܯ?0"c
7gradient_tape/model_1/conv1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterP?0?????!? :????0"i
>gradient_tape/model_1/res5a_branch1/Conv2D/Conv2DBackpropInputConv2DBackpropInput???p.???!ʒ?[??0"a
7gradient_tape/model_1/bn2a_branch1/FusedBatchNormGradV3FusedBatchNormGradV3?*V???!u??F?Z??"b
8gradient_tape/model_1/bn2a_branch2c/FusedBatchNormGradV3FusedBatchNormGradV3D???!???8?V??"b
8gradient_tape/model_1/bn2b_branch2c/FusedBatchNormGradV3FusedBatchNormGradV3gŝӠ??!??E?P??Q      Y@YIW?0???a??=?{?X@q????][??y????@8C?"?	
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Maxwell)(: B 