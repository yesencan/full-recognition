?	?+?Pϭi@?+?Pϭi@!?+?Pϭi@	f????f????!f????"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?+?Pϭi@~7ݲC???1??
?Th@I?f/C"@Y?,&6???r0*	?Q??A2f
/Iterator::Root::Prefetch::FlatMap[0]::Generator???V}d@!?I??	?X@)???V}d@1?I??	?X@:Preprocessing2E
Iterator::Root??Z???!ӭq5? ??)??8՚?1??x?c]??:Preprocessing2O
Iterator::Root::Prefetch?25	ސ??!???;_???)?25	ސ??1???;_???:Preprocessing2X
!Iterator::Root::Prefetch::FlatMapx*??g}d@!?????X@)y=??`?1;?2_6FT?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?4.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9f????I ?}*ߋ@Q١?=?W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	~7ݲC???~7ݲC???!~7ݲC???      ??!       "	??
?Th@??
?Th@!??
?Th@*      ??!       2      ??!       :	?f/C"@?f/C"@!?f/C"@B      ??!       J	?,&6????,&6???!?,&6???R      ??!       Z	?,&6????,&6???!?,&6???b      ??!       JGPUYf????b q ?}*ߋ@y١?=?W@?"a
6gradient_tape/model_1/conv0/Conv2D/Conv2DBackpropInputConv2DBackpropInput???<???!???<???0"p
Dgradient_tape/model_1/stage4_unit2_conv2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter=4)r???!΀>uf??0"p
Dgradient_tape/model_1/stage4_unit2_conv1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Āu??!l?2?{P??0"p
Dgradient_tape/model_1/stage4_unit1_conv2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterd??+?C??!??e???0"c
7gradient_tape/model_1/conv0/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter^.?????!?ͯ?????0"n
Cgradient_tape/model_1/stage4_unit2_conv1/Conv2D/Conv2DBackpropInputConv2DBackpropInputj???F???![^ŋz???0"n
Cgradient_tape/model_1/stage4_unit2_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput??????!.??\???0"n
Cgradient_tape/model_1/stage4_unit1_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput??E?"???!#J"'A~??0"X
.gradient_tape/model_1/bn0/FusedBatchNormGradV3FusedBatchNormGradV3?I??????!TV~???"?
!model_1/stage4_unit2_conv2/Conv2DConv2D?:*?a??!X?̑?N??0Q      Y@Y???i?`@a:X???X@qd?E?\N??yV?U???Y?"?

device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Maxwell)(: B 