	?G??钀@?G??钀@!?G??钀@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?G??钀@⏢?ܣ@1&?fe?Z?@A9`W?????Ix'??r??rEagerKernelExecute 0*	???M<?A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???Aoe@!s?9?X@)???Aoe@1s?9?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?($??;??!?+??u??)?($??;??1?+??u??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???????!????\???)?F?@??1Q?=A??:Preprocessing2F
Iterator::Modelo?????!?S??F???)?Oqn?1?F??M?a?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapJ?GWoe@!?;4?X@)??0Xre?1f???Y?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI m?ա/??QLƩxA?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	⏢?ܣ@⏢?ܣ@!⏢?ܣ@      ??!       "	&?fe?Z?@&?fe?Z?@!&?fe?Z?@*      ??!       2	9`W?????9`W?????!9`W?????:	x'??r??x'??r??!x'??r??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q m?ա/??yLƩxA?X@