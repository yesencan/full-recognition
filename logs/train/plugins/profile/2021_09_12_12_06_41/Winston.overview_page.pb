?	???6?h@???6?h@!???6?h@	????????????!??????"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???6?h@M?x$^???1?B=}Wh@I??J?.???YGx$(??r0*	???x??A2f
/Iterator::Root::Prefetch::FlatMap[0]::Generatorp?'v9h@!????[?X@)p?'v9h@1????[?X@:Preprocessing2O
Iterator::Root::Prefetch,?`p???!!Րȕ?),?`p???1!Րȕ?:Preprocessing2E
Iterator::Root?j+??ݳ?!8q{?=~??)??{b???1Q????3??:Preprocessing2X
!Iterator::Root::Prefetch::FlatMap??t??9h@!?C8p?X@)?&OYM?c?1XL?i?wT?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??????I?F???Q ???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	M?x$^???M?x$^???!M?x$^???      ??!       "	?B=}Wh@?B=}Wh@!?B=}Wh@*      ??!       2      ??!       :	??J?.?????J?.???!??J?.???B      ??!       J	Gx$(??Gx$(??!Gx$(??R      ??!       Z	Gx$(??Gx$(??!Gx$(??b      ??!       JGPUY??????b q?F???y ???X@?"a
6gradient_tape/model_1/conv0/Conv2D/Conv2DBackpropInputConv2DBackpropInputM:4?P??!M:4?P??0"p
Dgradient_tape/model_1/stage4_unit1_conv2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??;W???!???%?P??0"p
Dgradient_tape/model_1/stage4_unit2_conv1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?P?k??! "vC??0"p
Dgradient_tape/model_1/stage4_unit2_conv2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????>??!=!DӾ?0"n
Cgradient_tape/model_1/stage4_unit2_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?o??@???!s?㵶???0"c
7gradient_tape/model_1/conv0/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??k????!??`C???0"n
Cgradient_tape/model_1/stage4_unit2_conv1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?]3????!?q?????0"n
Cgradient_tape/model_1/stage4_unit1_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput? ??ȗ?!?Q?Z???0"X
.gradient_tape/model_1/bn0/FusedBatchNormGradV3FusedBatchNormGradV3:??´???!????A??"?
!model_1/stage4_unit1_conv2/Conv2DConv2D?6ԣ?P??!f??#?e??0Q      Y@Y???i?`@a:X???X@q ???$???y??94??Y?"?	
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