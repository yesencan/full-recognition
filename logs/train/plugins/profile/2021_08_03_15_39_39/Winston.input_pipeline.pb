	???HᏀ@???HᏀ@!???HᏀ@      ??!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:???HᏀ@=ڨNG@1???b?^?@IE?????rEagerKernelExecute 0*	ObXs?A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?U?p	d@!??@wl?X@)?U?p	d@1??@wl?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchIV?F??!?X?J?ɋ?)IV?F??1?X?J?ɋ?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??T2 T??!??"zo???)?x!?1???R?~?:Preprocessing2F
Iterator::Model??#???!?p-?n???)cz?(k?1m?V`??`?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap"nN%?	d@!)?I??X@)A?vc?1?_G???W?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI??䚅??Q??o???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	=ڨNG@=ڨNG@!=ڨNG@      ??!       "	???b?^?@???b?^?@!???b?^?@*      ??!       2      ??!       :	E?????E?????!E?????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??䚅??y??o???X@