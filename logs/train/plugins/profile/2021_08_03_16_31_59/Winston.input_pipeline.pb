	\W?o??@\W?o??@!\W?o??@      ??!       "?
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
	#h?$?U@#h?$?U@!#h?$?U@      ??!       "	?릔?W?@?릔?W?@!?릔?W?@*      ??!       2	Ĳ?CR??Ĳ?CR??!Ĳ?CR??:	?.l?V?	@?.l?V?	@!?.l?V?	@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@??I?a??y?ْy?X@