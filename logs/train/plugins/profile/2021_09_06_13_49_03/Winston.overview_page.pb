?	?1 ??@?1 ??@!?1 ??@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?1 ??@4?????@1?rh?mX?@AI??Q,???I?{L?4@rEagerKernelExecute 0*	    @MA2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator8???d?f@!?cF|??X@)8???d?f@1?cF|??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismLl>???!???3????)?)??F???1%  ?sM??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchɮ???{??!????????)ɮ???{??1????????:Preprocessing2F
Iterator::ModelZ?1?	ڬ?!?NԚ???)? 3??Ol?1%?E:?^?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapi?-x?f@!?R??X@)??1ZGUc?1!?pZ*U?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI ???QI??Q????ڒX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	4?????@4?????@!4?????@      ??!       "	?rh?mX?@?rh?mX?@!?rh?mX?@*      ??!       2	I??Q,???I??Q,???!I??Q,???:	?{L?4@?{L?4@!?{L?4@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q ???QI??y????ڒX@?"k
?gradient_tape/model_1/res5a_branch1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??	W7k??!??	W7k??0"l
@gradient_tape/model_1/res5c_branch2b/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter, E?l???!?N??mʜ?0"l
@gradient_tape/model_1/res5b_branch2b/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterB???/???!?f?΂??0"l
@gradient_tape/model_1/res5a_branch2b/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter_???m??!b?q/????0"c
7gradient_tape/model_1/conv1/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltera :ㅠ??!zh@?W??0"i
>gradient_tape/model_1/res5a_branch1/Conv2D/Conv2DBackpropInputConv2DBackpropInputҺ:?????!????A_??0"b
8gradient_tape/model_1/bn2c_branch2c/FusedBatchNormGradV3FusedBatchNormGradV3?o?8???!?R2?:]??"b
8gradient_tape/model_1/bn2b_branch2c/FusedBatchNormGradV3FusedBatchNormGradV3?????!.?CنY??"b
8gradient_tape/model_1/bn2a_branch2c/FusedBatchNormGradV3FusedBatchNormGradV3?V}??!???T??"a
7gradient_tape/model_1/bn2a_branch1/FusedBatchNormGradV3FusedBatchNormGradV3?&?q??!G?PN??Q      Y@YIW?0???a??=?{?X@q?T?0B7??y?`?pGC?"?	
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