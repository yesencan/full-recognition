	?1 ??@?1 ??@!?1 ??@      ??!       "?
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
	4?????@4?????@!4?????@      ??!       "	?rh?mX?@?rh?mX?@!?rh?mX?@*      ??!       2	I??Q,???I??Q,???!I??Q,???:	?{L?4@?{L?4@!?{L?4@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q ???QI??y????ڒX@