	ȗP???@ȗP???@!ȗP???@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCȗP???@?-z?@1???e?@A?? ???I???W:?@rEagerKernelExecute 0*	???K??A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator]???Oi@!P=??X@)]???Oi@1P=??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???R???!???5????)?ѫJC??1?I?} ???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch<hv?[???!U??MB??)<hv?[???1U??MB??:Preprocessing2F
Iterator::Model?hW!?'??!?]b?ʜ?)J]2???q?1Z?un??a?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapX:??Oi@!???^3?X@)?i?WV?d?1?J??
XT?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?Ig푛??Q?bJ???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?-z?@?-z?@!?-z?@      ??!       "	???e?@???e?@!???e?@*      ??!       2	?? ????? ???!?? ???:	???W:?@???W:?@!???W:?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?Ig푛??y?bJ???X@