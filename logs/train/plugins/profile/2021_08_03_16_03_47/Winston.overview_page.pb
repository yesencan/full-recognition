?	???J??@???J??@!???J??@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???J??@k?ѯ?@1_@/?9]?@Aݱ?&???I???_??rEagerKernelExecute 0*	B`???}A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator)?	??c@!? 5y??X@))?	??c@1? 5y??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?&????!	l?@+??)?&????1	l?@+??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism=Fy?尫?!??2??W??)???~1??1Wq?????:Preprocessing2F
Iterator::Model?#?\??!??)?kc??)??-??j?16nu??`?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapHG???c@!??????X@)????e?1y??M[?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI@tc?}???Q/rv??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	k?ѯ?@k?ѯ?@!k?ѯ?@      ??!       "	_@/?9]?@_@/?9]?@!_@/?9]?@*      ??!       2	ݱ?&???ݱ?&???!ݱ?&???:	???_?????_??!???_??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@tc?}???y/rv??X@?"k
?gradient_tape/model_1/res5a_branch1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterrw?????!rw?????0"l
@gradient_tape/model_1/res5c_branch2b/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?V_?̆?!?"????0"l
@gradient_tape/model_1/res5b_branch2b/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?w1C???!?uUՖ%??0"l
@gradient_tape/model_1/res5a_branch2b/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??u?S???!??2??̩?0"c
7gradient_tape/model_1/conv1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??p?I???!?O;?t??0"i
>gradient_tape/model_1/res5a_branch1/Conv2D/Conv2DBackpropInputConv2DBackpropInputU??e`??!pD(?kf??0"k
?gradient_tape/model_1/res4a_branch1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterEѓ??!??9Uc??0"b
8gradient_tape/model_1/bn2c_branch2c/FusedBatchNormGradV3FusedBatchNormGradV3??#4K??!?{?i_??"b
8gradient_tape/model_1/bn2b_branch2c/FusedBatchNormGradV3FusedBatchNormGradV3d?????!??c1?Z??"a
7gradient_tape/model_1/bn2a_branch1/FusedBatchNormGradV3FusedBatchNormGradV3˞?̘?!???gT??Q      Y@YIW?0???a??=?{?X@q?IEk???y,c???C?"?	
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