	???6?h@???6?h@!???6?h@	????????????!??????"q
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
	M?x$^???M?x$^???!M?x$^???      ??!       "	?B=}Wh@?B=}Wh@!?B=}Wh@*      ??!       2      ??!       :	??J?.?????J?.???!??J?.???B      ??!       J	Gx$(??Gx$(??!Gx$(??R      ??!       Z	Gx$(??Gx$(??!Gx$(??b      ??!       JGPUY??????b q?F???y ???X@