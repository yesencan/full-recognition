	???w??@???w??@!???w??@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???w??@uXᖏ?@1??.4?b?@A4??<???I?x@?t@rEagerKernelExecute 0*	,?فIA2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator p??s?e@!V?u???X@) p??s?e@1V?u???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchW횐???!GU?Ǔ??)W횐???1GU?Ǔ??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism	?<??t??!???ŞW??)?҇.?o??1-2v?+,}?:Preprocessing2F
Iterator::Modelfٓ????!?Qk	??)?uoEb?j?1?@
Ag^?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapBZcЉ?e@!?????X@)?(B?v?e?1??k]0Y?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?I??????Q?Z?9 ?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	uXᖏ?@uXᖏ?@!uXᖏ?@      ??!       "	??.4?b?@??.4?b?@!??.4?b?@*      ??!       2	4??<???4??<???!4??<???:	?x@?t@?x@?t@!?x@?t@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?I??????y?Z?9 ?X@