?	?\?].??@?\?].??@!?\?].??@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?\?].??@G?ҿ$5@1??D?M?@AZbe4?y??I???8?~@rEagerKernelExecute 0*	4^?I/QA2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator5A?} ?d@!,L?L??X@)5A?} ?d@1,L?L??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism_`V(????!в??6k??)$???+??1???
?n??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchv??ң??!JO[jώ?)v??ң??1JO[jώ?:Preprocessing2F
Iterator::Model׆?q???!7???5???)???p?1mVd???c?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapsI?v?d@!??M??X@)#?=?b?1t?T_j?V?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI@GW??s??Q??b,1?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	G?ҿ$5@G?ҿ$5@!G?ҿ$5@      ??!       "	??D?M?@??D?M?@!??D?M?@*      ??!       2	Zbe4?y??Zbe4?y??!Zbe4?y??:	???8?~@???8?~@!???8?~@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@GW??s??y??b,1?X@?"k
?gradient_tape/model_1/res5a_branch1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??ϱ-???!??ϱ-???0"l
@gradient_tape/model_1/res5a_branch2b/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????a???!?8????0"l
@gradient_tape/model_1/res5c_branch2b/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??(?????!?A?P0??0"l
@gradient_tape/model_1/res5b_branch2b/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????U???!?wFa&ԩ?0"c
7gradient_tape/model_1/conv1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?J?̞???!w?kN|??0"i
>gradient_tape/model_1/res5a_branch1/Conv2D/Conv2DBackpropInputConv2DBackpropInput8~??Qŀ?!?@?V??0"b
8gradient_tape/model_1/bn2a_branch2c/FusedBatchNormGradV3FusedBatchNormGradV3???2k??!t%`?^X??"a
7gradient_tape/model_1/bn2a_branch1/FusedBatchNormGradV3FusedBatchNormGradV32?B????!R??:V??"b
8gradient_tape/model_1/bn2b_branch2c/FusedBatchNormGradV3FusedBatchNormGradV3(???1??!z???mS??"b
8gradient_tape/model_1/bn2c_branch2c/FusedBatchNormGradV3FusedBatchNormGradV3̼?2??!;}??P??Q      Y@YIW?0???a??=?{?X@qT-?\}??ya?ϛ C?"?	
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