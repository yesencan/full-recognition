	?m?(蚀@?m?(蚀@!?m?(蚀@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?m?(蚀@?<+i?g@1?&??`?@A???+f??I<?(Aa??rEagerKernelExecute 0*	?$?*?A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?J??^c@!֏??R?X@)?J??^c@1֏??R?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch(I?L?ٖ?!?.4?}???)(I?L?ٖ?1?.4?}???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism2?g@???!????>??)zލ?A??1>?c"????:Preprocessing2F
Iterator::Model??2SZ??!3?!΋??)?9??!l?1????mb?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???qnc@!?Cg?X@)(??ȯ_?1f?
?#?T?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI mNGI??QL??ڮ?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?<+i?g@?<+i?g@!?<+i?g@      ??!       "	?&??`?@?&??`?@!?&??`?@*      ??!       2	???+f?????+f??!???+f??:	<?(Aa??<?(Aa??!<?(Aa??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q mNGI??yL??ڮ?X@