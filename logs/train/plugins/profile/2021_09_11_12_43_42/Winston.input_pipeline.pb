	?+?Pϭi@?+?Pϭi@!?+?Pϭi@	f????f????!f????"q
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
?Th@*      ??!       2      ??!       :	?f/C"@?f/C"@!?f/C"@B      ??!       J	?,&6????,&6???!?,&6???R      ??!       Z	?,&6????,&6???!?,&6???b      ??!       JGPUYf????b q ?}*ߋ@y١?=?W@