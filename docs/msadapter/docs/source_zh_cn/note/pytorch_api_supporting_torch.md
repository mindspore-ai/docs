# torch

## Tensor

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.is_tensor](https://pytorch.org/docs/2.1/generated/torch.is_tensor.html)|Not Support|N/A|
|[torch.is_storage](https://pytorch.org/docs/2.1/generated/torch.is_storage.html)|Not Support|N/A|
|[torch.is_complex](https://pytorch.org/docs/2.1/generated/torch.is_complex.html)|Not Support|N/A|
|[torch.is_conj](https://pytorch.org/docs/2.1/generated/torch.is_conj.html)|Not Support|N/A|
|[torch.is_floating_point](https://pytorch.org/docs/2.1/generated/torch.is_floating_point.html)|Beta|N/A|
|[torch.is_nonzero](https://pytorch.org/docs/2.1/generated/torch.is_nonzero.html)|Not Support|N/A|
|[torch.set_default_dtype](https://pytorch.org/docs/2.1/generated/torch.set_default_dtype.html)|Not Support|N/A|
|[torch.get_default_dtype](https://pytorch.org/docs/2.1/generated/torch.get_default_dtype.html)|Not Support|N/A|
|[torch.set_default_device](https://pytorch.org/docs/2.1/generated/torch.set_default_device.html)|Not Support|N/A|
|[torch.get_default_device](https://pytorch.org/docs/2.1/generated/torch.get_default_device.html)|Not Support|N/A|
|[torch.set_default_tensor_type](https://pytorch.org/docs/2.1/generated/torch.set_default_tensor_type.html)|Not Support|N/A|
|[torch.numel](https://pytorch.org/docs/2.1/generated/torch.numel.html)|Beta|N/A|
|[torch.set_printoptions](https://pytorch.org/docs/2.1/generated/torch.set_printoptions.html)|Not Support|N/A|
|[torch.set_flush_denormal](https://pytorch.org/docs/2.1/generated/torch.set_flush_denormal.html)|Not Support|N/A|

### Creation Ops

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.tensor](https://pytorch.org/docs/2.1/generated/torch.tensor.html)|Not Support|N/A|
|[torch.sparse_coo_tensor](https://pytorch.org/docs/2.1/generated/torch.sparse_coo_tensor.html)|Not Support|N/A|
|[torch.sparse_csr_tensor](https://pytorch.org/docs/2.1/generated/torch.sparse_csr_tensor.html)|Not Support|N/A|
|[torch.sparse_csc_tensor](https://pytorch.org/docs/2.1/generated/torch.sparse_csc_tensor.html)|Not Support|N/A|
|[torch.sparse_bsr_tensor](https://pytorch.org/docs/2.1/generated/torch.sparse_bsr_tensor.html)|Not Support|N/A|
|[torch.sparse_bsc_tensor](https://pytorch.org/docs/2.1/generated/torch.sparse_bsc_tensor.html)|Not Support|N/A|
|[torch.asarray](https://pytorch.org/docs/2.1/generated/torch.asarray.html)|Not Support|N/A|
|[torch.as_tensor](https://pytorch.org/docs/2.1/generated/torch.as_tensor.html)|Not Support|N/A|
|[torch.as_strided](https://pytorch.org/docs/2.1/generated/torch.as_strided.html)|Beta|N/A|
|[torch.from_file](https://pytorch.org/docs/2.1/generated/torch.from_file.html)|Not Support|N/A|
|[torch.from_numpy](https://pytorch.org/docs/2.1/generated/torch.from_numpy.html)|Not Support|N/A|
|[torch.from_dlpack](https://pytorch.org/docs/2.1/generated/torch.from_dlpack.html)|Not Support|N/A|
|[torch.frombuffer](https://pytorch.org/docs/2.1/generated/torch.frombuffer.html)|Not Support|N/A|
|[torch.zeros](https://pytorch.org/docs/2.1/generated/torch.zeros.html)|Beta|不支持out、layout参数|
|[torch.zeros_like](https://pytorch.org/docs/2.1/generated/torch.zeros_like.html)|Beta|不支持layout、device、requires_grad参数|
|[torch.ones](https://pytorch.org/docs/2.1/generated/torch.ones.html)|Beta|不支持out、layout、requires_grad参数|
|[torch.ones_like](https://pytorch.org/docs/2.1/generated/torch.ones_like.html)|Beta|不支持layout、requires_grad、memory_format参数|
|[torch.arange](https://pytorch.org/docs/2.1/generated/torch.arange.html)|Beta|不支持out、lay_out、requires_grad参数|
|[torch.range](https://pytorch.org/docs/2.1/generated/torch.range.html)|Beta|不支持out、requires_grad参数|
|[torch.linspace](https://pytorch.org/docs/2.1/generated/torch.linspace.html)|Beta|不支持out、layout、device、requires_grad参数|
|[torch.logspace](https://pytorch.org/docs/2.1/generated/torch.logspace.html)|Beta|不支持out出参|
|[torch.eye](https://pytorch.org/docs/2.1/generated/torch.eye.html)|Beta|不支持out、layout、device、requires_grad参数|
|[torch.empty](https://pytorch.org/docs/2.1/generated/torch.empty.html)|Beta|不支持out出参、pin_memory、memory_format参数|
|[torch.empty_like](https://pytorch.org/docs/2.1/generated/torch.empty_like.html)|Beta|N/A|
|[torch.empty_strided](https://pytorch.org/docs/2.1/generated/torch.empty_strided.html)|Not Support|N/A|
|[torch.full](https://pytorch.org/docs/2.1/generated/torch.full.html)|Beta|不支持out、layout、device、require_grad参数|
|[torch.full_like](https://pytorch.org/docs/2.1/generated/torch.full_like.html)|Beta|不支持out、layout、device、require_grad参数|
|[torch.quantize_per_tensor](https://pytorch.org/docs/2.1/generated/torch.quantize_per_tensor.html)|Not Support|N/A|
|[torch.quantize_per_channel](https://pytorch.org/docs/2.1/generated/torch.quantize_per_channel.html)|Not Support|N/A|
|[torch.dequantize](https://pytorch.org/docs/2.1/generated/torch.dequantize.html)|Not Support|N/A|
|[torch.complex](https://pytorch.org/docs/2.1/generated/torch.complex.html)|Not Support|N/A|
|[torch.polar](https://pytorch.org/docs/2.1/generated/torch.polar.html)|Beta|不支持out出参|
|[torch.heaviside](https://pytorch.org/docs/2.1/generated/torch.heaviside.html)|Beta|不支持out出参|

## Indexing, Slicing, Joining, Mutation Ops

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.adjoint](https://pytorch.org/docs/2.1/generated/torch.adjoint.html)|Not Support|N/A|
|[torch.argwhere](https://pytorch.org/docs/2.1/generated/torch.argwhere.html)|Beta|N/A|
|[torch.cat](https://pytorch.org/docs/2.1/generated/torch.cat.html)|Stable|N/A|
|[torch.concat](https://pytorch.org/docs/2.1/generated/torch.concat.html)|Stable|N/A|
|[torch.concatenate](https://pytorch.org/docs/2.1/generated/torch.concatenate.html)|Stable|N/A|
|[torch.conj](https://pytorch.org/docs/2.1/generated/torch.conj.html)|Beta|N/A|
|[torch.chunk](https://pytorch.org/docs/2.1/generated/torch.chunk.html)|Stable|N/A|
|[torch.dsplit](https://pytorch.org/docs/2.1/generated/torch.dsplit.html)|Not Support|N/A|
|[torch.column_stack](https://pytorch.org/docs/2.1/generated/torch.column_stack.html)|Not Support|N/A|
|[torch.dstack](https://pytorch.org/docs/2.1/generated/torch.dstack.html)|Not Support|N/A|
|[torch.gather](https://pytorch.org/docs/2.1/generated/torch.gather.html)|Beta|不支持out、layout、sparse_grad参数|
|[torch.hsplit](https://pytorch.org/docs/2.1/generated/torch.hsplit.html)|Not Support|N/A|
|[torch.hstack](https://pytorch.org/docs/2.1/generated/torch.hstack.html)|Beta|不支持out出参|
|[torch.index_add](https://pytorch.org/docs/2.1/generated/torch.index_add.html)|Beta|不支持out出参|
|[torch.index_copy](https://pytorch.org/docs/2.1/generated/torch.index_copy.html)|Not Support|N/A|
|[torch.index_reduce](https://pytorch.org/docs/2.1/generated/torch.index_reduce.html)|Not Support|N/A|
|[torch.index_select](https://pytorch.org/docs/2.1/generated/torch.index_select.html)|Stable|N/A|
|[torch.masked_select](https://pytorch.org/docs/2.1/generated/torch.masked_select.html)|Stable|N/A|
|[torch.movedim](https://pytorch.org/docs/2.1/generated/torch.movedim.html)|Not Support|N/A|
|[torch.moveaxis](https://pytorch.org/docs/2.1/generated/torch.moveaxis.html)|Beta|N/A|
|[torch.narrow](https://pytorch.org/docs/2.1/generated/torch.narrow.html)|Stable|N/A|
|[torch.narrow_copy](https://pytorch.org/docs/2.1/generated/torch.narrow_copy.html)|Not Support|N/A|
|[torch.nonzero](https://pytorch.org/docs/2.1/generated/torch.nonzero.html)|Beta|不支持out出参|
|[torch.permute](https://pytorch.org/docs/2.1/generated/torch.permute.html)|Stable|N/A|
|[torch.reshape](https://pytorch.org/docs/2.1/generated/torch.reshape.html)|Stable|N/A|
|[torch.row_stack](https://pytorch.org/docs/2.1/generated/torch.row_stack.html)|Not Support|N/A|
|[torch.select](https://pytorch.org/docs/2.1/generated/torch.select.html)|Stable|N/A|
|[torch.scatter](https://pytorch.org/docs/2.1/generated/torch.scatter.html)|Stable|N/A|
|[torch.diagonal_scatter](https://pytorch.org/docs/2.1/generated/torch.diagonal_scatter.html)|Not Support|N/A|
|[torch.select_scatter](https://pytorch.org/docs/2.1/generated/torch.select_scatter.html)|Not Support|N/A|
|[torch.slice_scatter](https://pytorch.org/docs/2.1/generated/torch.slice_scatter.html)|Not Support|N/A|
|[torch.scatter_add](https://pytorch.org/docs/2.1/generated/torch.scatter_add.html)|Stable|N/A|
|[torch.scatter_reduce](https://pytorch.org/docs/2.1/generated/torch.scatter_reduce.html)|Not Support|N/A|
|[torch.split](https://pytorch.org/docs/2.1/generated/torch.split.html)|Stable|N/A|
|[torch.squeeze](https://pytorch.org/docs/2.1/generated/torch.squeeze.html)|Beta|N/A|
|[torch.stack](https://pytorch.org/docs/2.1/generated/torch.stack.html)|Stable|N/A|
|[torch.swapaxes](https://pytorch.org/docs/2.1/generated/torch.swapaxes.html)|Beta|N/A|
|[torch.swapdims](https://pytorch.org/docs/2.1/generated/torch.swapdims.html)|Beta|N/A|
|[torch.t](https://pytorch.org/docs/2.1/generated/torch.t.html)|Not Support|N/A|
|[torch.take](https://pytorch.org/docs/2.1/generated/torch.take.html)|Beta|N/A|
|[torch.take_along_dim](https://pytorch.org/docs/2.1/generated/torch.take_along_dim.html)|Not Support|N/A|
|[torch.tensor_split](https://pytorch.org/docs/2.1/generated/torch.tensor_split.html)|Beta|N/A|
|[torch.tile](https://pytorch.org/docs/2.1/generated/torch.tile.html)|Stable|N/A|
|[torch.transpose](https://pytorch.org/docs/2.1/generated/torch.transpose.html)|Stable|N/A|
|[torch.unbind](https://pytorch.org/docs/2.1/generated/torch.unbind.html)|Beta|N/A|
|[torch.unravel_index](https://pytorch.org/docs/2.1/generated/torch.unravel_index.html)|Not Support|N/A|
|[torch.unsqueeze](https://pytorch.org/docs/2.1/generated/torch.unsqueeze.html)|Stable|N/A|
|[torch.vsplit](https://pytorch.org/docs/2.1/generated/torch.vsplit.html)|Not Support|N/A|
|[torch.vstack](https://pytorch.org/docs/2.1/generated/torch.vstack.html)|Beta|不支持out出参|
|[torch.where](https://pytorch.org/docs/2.1/generated/torch.where.html)|Stable|N/A|

## Accelerators

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.Stream](https://pytorch.org/docs/2.1/generated/torch.Stream.html)|Not Support|N/A|
|[torch.Event](https://pytorch.org/docs/2.1/generated/torch.Event.html)|Not Support|N/A|

## Generators

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.Generator](https://pytorch.org/docs/2.1/generated/torch.Generator.html)|Not Support|N/A|

## Random sampling

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.seed](https://pytorch.org/docs/2.1/generated/torch.seed.html)|Not Support|N/A|
|[torch.manual_seed](https://pytorch.org/docs/2.1/generated/torch.manual_seed.html)|Not Support|N/A|
|[torch.initial_seed](https://pytorch.org/docs/2.1/generated/torch.initial_seed.html)|Not Support|N/A|
|[torch.get_rng_state](https://pytorch.org/docs/2.1/generated/torch.get_rng_state.html)|Not Support|N/A|
|[torch.set_rng_state](https://pytorch.org/docs/2.1/generated/torch.set_rng_state.html)|Not Support|N/A|
|[torch.bernoulli](https://pytorch.org/docs/2.1/generated/torch.bernoulli.html)|Stable|N/A|
|[torch.multinomial](https://pytorch.org/docs/2.1/generated/torch.multinomial.html)|Beta|不支持out出参|
|[torch.normal](https://pytorch.org/docs/2.1/generated/torch.normal.html)|Beta|不支持generator、out参数|
|[torch.poisson](https://pytorch.org/docs/2.1/generated/torch.poisson.html)|Not Support|N/A|
|[torch.rand](https://pytorch.org/docs/2.1/generated/torch.rand.html)|Beta|不支持generator、out、layout、requires_grad参数|
|[torch.rand_like](https://pytorch.org/docs/2.1/generated/torch.rand_like.html)|Beta|不支持layout、device、requires_grad、memory_format参数|
|[torch.randint](https://pytorch.org/docs/2.1/generated/torch.randint.html)|Stable|N/A|
|[torch.randint_like](https://pytorch.org/docs/2.1/generated/torch.randint_like.html)|Not Support|N/A|
|[torch.randn](https://pytorch.org/docs/2.1/generated/torch.randn.html)|Beta|入参不一致|
|[torch.randn_like](https://pytorch.org/docs/2.1/generated/torch.randn_like.html)|Beta|不支持layout、device、requires_grad、memory_format参数|
|[torch.randperm](https://pytorch.org/docs/2.1/generated/torch.randperm.html)|Beta|不支持out、layout、device、requires_grad、pin_memory参数|

### In-place random sampling

暂不支持

### Quasi-random sampling

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.quasirandom.SobolEngine](https://pytorch.org/docs/2.1/generated/torch.quasirandom.SobolEngine.html)|Not Support|N/A|

## Serialization

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.save](https://pytorch.org/docs/2.1/generated/torch.save.html)|Not Support|N/A|
|[torch.load](https://pytorch.org/docs/2.1/generated/torch.load.html)|Not Support|N/A|

## Parallelism

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.get_num_threads](https://pytorch.org/docs/2.1/generated/torch.get_num_threads.html)|Not Support|N/A|
|[torch.set_num_threads](https://pytorch.org/docs/2.1/generated/torch.set_num_threads.html)|Not Support|N/A|
|[torch.get_num_interop_threads](https://pytorch.org/docs/2.1/generated/torch.get_num_interop_threads.html)|Not Support|N/A|
|[torch.set_num_interop_threads](https://pytorch.org/docs/2.1/generated/torch.set_num_interop_threads.html)|Not Support|N/A|

## Locally disabling gradient computation

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.no_grad](https://pytorch.org/docs/2.1/generated/torch.no_grad.html)|Not Support|N/A|
|[torch.enable_grad](https://pytorch.org/docs/2.1/generated/torch.enable_grad.html)|Not Support|N/A|
|[torch.set_grad_enabled](https://pytorch.org/docs/2.1/generated/torch.set_grad_enabled.html)|Not Support|N/A|
|[torch.is_grad_enabled](https://pytorch.org/docs/2.1/generated/torch.is_grad_enabled.html)|Not Support|N/A|
|[torch.inference_mode](https://pytorch.org/docs/2.1/generated/torch.inference_mode.html)|Not Support|N/A|
|[torch.is_inference_mode_enabled](https://pytorch.org/docs/2.1/generated/torch.is_inference_mode_enabled.html)|Not Support|N/A|

## Math operations

### Constants

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.inf](https://pytorch.org/docs/2.1/generated/torch.inf.html)|Not Support|N/A|
|[torch.nan](https://pytorch.org/docs/2.1/generated/torch.nan.html)|Not Support|N/A|

### Pointwise Ops

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.abs](https://pytorch.org/docs/2.1/generated/torch.abs.html)|Stable|N/A|
|[torch.absolute](https://pytorch.org/docs/2.1/generated/torch.absolute.html)|Stable|N/A|
|[torch.acos](https://pytorch.org/docs/2.1/generated/torch.acos.html)|Stable|N/A|
|[torch.arccos](https://pytorch.org/docs/2.1/generated/torch.arccos.html)|Not Support|N/A|
|[torch.acosh](https://pytorch.org/docs/2.1/generated/torch.acosh.html)|Stable|N/A|
|[torch.arccosh](https://pytorch.org/docs/2.1/generated/torch.arccosh.html)|Beta|不支持out出参|
|[torch.add](https://pytorch.org/docs/2.1/generated/torch.add.html)|Stable|N/A|
|[torch.addcdiv](https://pytorch.org/docs/2.1/generated/torch.addcdiv.html)|Beta|不支持out出参|
|[torch.addcmul](https://pytorch.org/docs/2.1/generated/torch.addcmul.html)|Beta|不支持out出参|
|[torch.angle](https://pytorch.org/docs/2.1/generated/torch.angle.html)|Beta|不支持out出参|
|[torch.asin](https://pytorch.org/docs/2.1/generated/torch.asin.html)|Stable|N/A|
|[torch.arcsin](https://pytorch.org/docs/2.1/generated/torch.arcsin.html)|Stable|N/A|
|[torch.asinh](https://pytorch.org/docs/2.1/generated/torch.asinh.html)|Stable|N/A|
|[torch.arcsinh](https://pytorch.org/docs/2.1/generated/torch.arcsinh.html)|Stable|N/A|
|[torch.atan](https://pytorch.org/docs/2.1/generated/torch.atan.html)|Stable|N/A|
|[torch.arctan](https://pytorch.org/docs/2.1/generated/torch.arctan.html)|Stable|N/A|
|[torch.atanh](https://pytorch.org/docs/2.1/generated/torch.atanh.html)|Stable|N/A|
|[torch.arctanh](https://pytorch.org/docs/2.1/generated/torch.arctanh.html)|Stable|N/A|
|[torch.atan2](https://pytorch.org/docs/2.1/generated/torch.atan2.html)|Stable|N/A|
|[torch.arctan2](https://pytorch.org/docs/2.1/generated/torch.arctan2.html)|Stable|N/A|
|[torch.bitwise_not](https://pytorch.org/docs/2.1/generated/torch.bitwise_not.html)|Not Support|N/A|
|[torch.bitwise_and](https://pytorch.org/docs/2.1/generated/torch.bitwise_and.html)|Stable|N/A|
|[torch.bitwise_or](https://pytorch.org/docs/2.1/generated/torch.bitwise_or.html)|Stable|N/A|
|[torch.bitwise_xor](https://pytorch.org/docs/2.1/generated/torch.bitwise_xor.html)|Stable|N/A|
|[torch.bitwise_left_shift](https://pytorch.org/docs/2.1/generated/torch.bitwise_left_shift.html)|Beta|不支持out出参|
|[torch.bitwise_right_shift](https://pytorch.org/docs/2.1/generated/torch.bitwise_right_shift.html)|Beta|不支持out出参|
|[torch.ceil](https://pytorch.org/docs/2.1/generated/torch.ceil.html)|Stable|N/A|
|[torch.clamp](https://pytorch.org/docs/2.1/generated/torch.clamp.html)|Stable|N/A|
|[torch.clip](https://pytorch.org/docs/2.1/generated/torch.clip.html)|Beta|不支持out出参|
|[torch.conj_physical](https://pytorch.org/docs/2.1/generated/torch.conj_physical.html)|Not Support|N/A|
|[torch.copysign](https://pytorch.org/docs/2.1/generated/torch.copysign.html)|Not Support|N/A|
|[torch.cos](https://pytorch.org/docs/2.1/generated/torch.cos.html)|Stable|N/A|
|[torch.cosh](https://pytorch.org/docs/2.1/generated/torch.cosh.html)|Stable|N/A|
|[torch.deg2rad](https://pytorch.org/docs/2.1/generated/torch.deg2rad.html)|Beta|不支持out出参|
|[torch.div](https://pytorch.org/docs/2.1/generated/torch.div.html)|Stable|N/A|
|[torch.divide](https://pytorch.org/docs/2.1/generated/torch.divide.html)|Beta|不支持out出参|
|[torch.digamma](https://pytorch.org/docs/2.1/generated/torch.digamma.html)|Beta|不支持out出参|
|[torch.erf](https://pytorch.org/docs/2.1/generated/torch.erf.html)|Stable|N/A|
|[torch.erfc](https://pytorch.org/docs/2.1/generated/torch.erfc.html)|Stable|N/A|
|[torch.erfinv](https://pytorch.org/docs/2.1/generated/torch.erfinv.html)|Stable|N/A|
|[torch.exp](https://pytorch.org/docs/2.1/generated/torch.exp.html)|Stable|N/A|
|[torch.exp2](https://pytorch.org/docs/2.1/generated/torch.exp2.html)|Beta|不支持out出参|
|[torch.expm1](https://pytorch.org/docs/2.1/generated/torch.expm1.html)|Stable|N/A|
|[torch.fake_quantize_per_channel_affine](https://pytorch.org/docs/2.1/generated/torch.fake_quantize_per_channel_affine.html)|Not Support|N/A|
|[torch.fake_quantize_per_tensor_affine](https://pytorch.org/docs/2.1/generated/torch.fake_quantize_per_tensor_affine.html)|Not Support|N/A|
|[torch.fix](https://pytorch.org/docs/2.1/generated/torch.fix.html)|Not Support|N/A|
|[torch.float_power](https://pytorch.org/docs/2.1/generated/torch.float_power.html)|Beta|不支持out出参|
|[torch.floor](https://pytorch.org/docs/2.1/generated/torch.floor.html)|Stable|N/A|
|[torch.floor_divide](https://pytorch.org/docs/2.1/generated/torch.floor_divide.html)|Beta|不支持out出参|
|[torch.fmod](https://pytorch.org/docs/2.1/generated/torch.fmod.html)|Beta|不支持out出参|
|[torch.frac](https://pytorch.org/docs/2.1/generated/torch.frac.html)|Beta|不支持out出参|
|[torch.frexp](https://pytorch.org/docs/2.1/generated/torch.frexp.html)|Not Support|N/A|
|[torch.gradient](https://pytorch.org/docs/2.1/generated/torch.gradient.html)|Not Support|N/A|
|[torch.imag](https://pytorch.org/docs/2.1/generated/torch.imag.html)|Beta|N/A|
|[torch.ldexp](https://pytorch.org/docs/2.1/generated/torch.ldexp.html)|Not Support|N/A|
|[torch.lerp](https://pytorch.org/docs/2.1/generated/torch.lerp.html)|Beta|不支持out出参|
|[torch.lgamma](https://pytorch.org/docs/2.1/generated/torch.lgamma.html)|Beta|不支持out出参|
|[torch.log](https://pytorch.org/docs/2.1/generated/torch.log.html)|Stable|N/A|
|[torch.log10](https://pytorch.org/docs/2.1/generated/torch.log10.html)|Beta|不支持out出参|
|[torch.log1p](https://pytorch.org/docs/2.1/generated/torch.log1p.html)|Stable|N/A|
|[torch.log2](https://pytorch.org/docs/2.1/generated/torch.log2.html)|Stable|N/A|
|[torch.logaddexp](https://pytorch.org/docs/2.1/generated/torch.logaddexp.html)|Not Support|N/A|
|[torch.logaddexp2](https://pytorch.org/docs/2.1/generated/torch.logaddexp2.html)|Not Support|N/A|
|[torch.logical_and](https://pytorch.org/docs/2.1/generated/torch.logical_and.html)|Stable|N/A|
|[torch.logical_not](https://pytorch.org/docs/2.1/generated/torch.logical_not.html)|Stable|N/A|
|[torch.logical_or](https://pytorch.org/docs/2.1/generated/torch.logical_or.html)|Stable|N/A|
|[torch.logical_xor](https://pytorch.org/docs/2.1/generated/torch.logical_xor.html)|Stable|N/A|
|[torch.logit](https://pytorch.org/docs/2.1/generated/torch.logit.html)|Beta|不支持out出参|
|[torch.hypot](https://pytorch.org/docs/2.1/generated/torch.hypot.html)|Beta|不支持out出参|
|[torch.i0](https://pytorch.org/docs/2.1/generated/torch.i0.html)|Not Support|N/A|
|[torch.igamma](https://pytorch.org/docs/2.1/generated/torch.igamma.html)|Beta|不支持out出参|
|[torch.igammac](https://pytorch.org/docs/2.1/generated/torch.igammac.html)|Beta|不支持out出参|
|[torch.mul](https://pytorch.org/docs/2.1/generated/torch.mul.html)|Stable|N/A|
|[torch.multiply](https://pytorch.org/docs/2.1/generated/torch.multiply.html)|Not Support|N/A|
|[torch.mvlgamma](https://pytorch.org/docs/2.1/generated/torch.mvlgamma.html)|Beta|不支持out出参|
|[torch.nan_to_num](https://pytorch.org/docs/2.1/generated/torch.nan_to_num.html)|Stable|N/A|
|[torch.neg](https://pytorch.org/docs/2.1/generated/torch.neg.html)|Stable|N/A|
|[torch.negative](https://pytorch.org/docs/2.1/generated/torch.negative.html)|Beta|不支持out出参|
|[torch.nextafter](https://pytorch.org/docs/2.1/generated/torch.nextafter.html)|Beta|不支持out出参|
|[torch.polygamma](https://pytorch.org/docs/2.1/generated/torch.polygamma.html)|Beta|不支持out出参|
|[torch.positive](https://pytorch.org/docs/2.1/generated/torch.positive.html)|Beta|N/A|
|[torch.pow](https://pytorch.org/docs/2.1/generated/torch.pow.html)|Stable|N/A|
|[torch.quantized_batch_norm](https://pytorch.org/docs/2.1/generated/torch.quantized_batch_norm.html)|Not Support|N/A|
|[torch.quantized_max_pool1d](https://pytorch.org/docs/2.1/generated/torch.quantized_max_pool1d.html)|Not Support|N/A|
|[torch.quantized_max_pool2d](https://pytorch.org/docs/2.1/generated/torch.quantized_max_pool2d.html)|Not Support|N/A|
|[torch.rad2deg](https://pytorch.org/docs/2.1/generated/torch.rad2deg.html)|Beta|不支持out出参|
|[torch.real](https://pytorch.org/docs/2.1/generated/torch.real.html)|Beta|N/A|
|[torch.reciprocal](https://pytorch.org/docs/2.1/generated/torch.reciprocal.html)|Stable|N/A|
|[torch.remainder](https://pytorch.org/docs/2.1/generated/torch.remainder.html)|Stable|N/A|
|[torch.round](https://pytorch.org/docs/2.1/generated/torch.round.html)|Beta|不支持out|
|[torch.rsqrt](https://pytorch.org/docs/2.1/generated/torch.rsqrt.html)|Stable|N/A|
|[torch.sigmoid](https://pytorch.org/docs/2.1/generated/torch.sigmoid.html)|Stable|N/A|
|[torch.sign](https://pytorch.org/docs/2.1/generated/torch.sign.html)|Stable|N/A|
|[torch.sgn](https://pytorch.org/docs/2.1/generated/torch.sgn.html)|Not Support|N/A|
|[torch.signbit](https://pytorch.org/docs/2.1/generated/torch.signbit.html)|Not Support|N/A|
|[torch.sin](https://pytorch.org/docs/2.1/generated/torch.sin.html)|Stable|N/A|
|[torch.sinc](https://pytorch.org/docs/2.1/generated/torch.sinc.html)|Stable|N/A|
|[torch.sinh](https://pytorch.org/docs/2.1/generated/torch.sinh.html)|Stable|N/A|
|[torch.softmax](https://pytorch.org/docs/2.1/generated/torch.softmax.html)|Stable|N/A|
|[torch.sqrt](https://pytorch.org/docs/2.1/generated/torch.sqrt.html)|Stable|N/A|
|[torch.square](https://pytorch.org/docs/2.1/generated/torch.square.html)|Stable|N/A|
|[torch.sub](https://pytorch.org/docs/2.1/generated/torch.sub.html)|Stable|N/A|
|[torch.subtract](https://pytorch.org/docs/2.1/generated/torch.subtract.html)|Beta|不支持out、alpha|
|[torch.tan](https://pytorch.org/docs/2.1/generated/torch.tan.html)|Stable|N/A|
|[torch.tanh](https://pytorch.org/docs/2.1/generated/torch.tanh.html)|Stable|N/A|
|[torch.true_divide](https://pytorch.org/docs/2.1/generated/torch.true_divide.html)|Beta|不支持out出参|
|[torch.trunc](https://pytorch.org/docs/2.1/generated/torch.trunc.html)|Stable|N/A|
|[torch.xlogy](https://pytorch.org/docs/2.1/generated/torch.xlogy.html)|Stable|N/A|

### Reduction Ops

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.argmax](https://pytorch.org/docs/2.1/generated/torch.argmax.html)|Stable|N/A|
|[torch.argmin](https://pytorch.org/docs/2.1/generated/torch.argmin.html)|Stable|N/A|
|[torch.amax](https://pytorch.org/docs/2.1/generated/torch.amax.html)|Beta|不支持out出参|
|[torch.amin](https://pytorch.org/docs/2.1/generated/torch.amin.html)|Beta|不支持out出参|
|[torch.aminmax](https://pytorch.org/docs/2.1/generated/torch.aminmax.html)|Beta|不支持out出参|
|[torch.all](https://pytorch.org/docs/2.1/generated/torch.all.html)|Stable|N/A|
|[torch.any](https://pytorch.org/docs/2.1/generated/torch.any.html)|Stable|dim入参具有默认值None|
|[torch.max](https://pytorch.org/docs/2.1/generated/torch.max.html)|Stable|N/A|
|[torch.min](https://pytorch.org/docs/2.1/generated/torch.min.html)|Stable|N/A|
|[torch.dist](https://pytorch.org/docs/2.1/generated/torch.dist.html)|Not Support|N/A|
|[torch.logsumexp](https://pytorch.org/docs/2.1/generated/torch.logsumexp.html)|Beta|不支持out出参|
|[torch.mean](https://pytorch.org/docs/2.1/generated/torch.mean.html)|Stable|N/A|
|[torch.nanmean](https://pytorch.org/docs/2.1/generated/torch.nanmean.html)|Beta|不支持out出参|
|[torch.median](https://pytorch.org/docs/2.1/generated/torch.median.html)|Stable|N/A|
|[torch.nanmedian](https://pytorch.org/docs/2.1/generated/torch.nanmedian.html)|Not Support|N/A|
|[torch.mode](https://pytorch.org/docs/2.1/generated/torch.mode.html)|Not Support|N/A|
|[torch.norm](https://pytorch.org/docs/2.1/generated/torch.norm.html)|Stable|N/A|
|[torch.nansum](https://pytorch.org/docs/2.1/generated/torch.nansum.html)|Beta|N/A|
|[torch.prod](https://pytorch.org/docs/2.1/generated/torch.prod.html)|Stable|N/A|
|[torch.quantile](https://pytorch.org/docs/2.1/generated/torch.quantile.html)|Beta|不支持out出参|
|[torch.nanquantile](https://pytorch.org/docs/2.1/generated/torch.nanquantile.html)|Beta|不支持out出参|
|[torch.std](https://pytorch.org/docs/2.1/generated/torch.std.html)|Beta|不支持out出参|
|[torch.std_mean](https://pytorch.org/docs/2.1/generated/torch.std_mean.html)|Beta|不支持out出参|
|[torch.sum](https://pytorch.org/docs/2.1/generated/torch.sum.html)|Beta|N/A|
|[torch.unique](https://pytorch.org/docs/2.1/generated/torch.unique.html)|Stable|N/A|
|[torch.unique_consecutive](https://pytorch.org/docs/2.1/generated/torch.unique_consecutive.html)|Beta|N/A|
|[torch.var](https://pytorch.org/docs/2.1/generated/torch.var.html)|Beta|不支持out出参|
|[torch.var_mean](https://pytorch.org/docs/2.1/generated/torch.var_mean.html)|Beta|不支持out出参|
|[torch.count_nonzero](https://pytorch.org/docs/2.1/generated/torch.count_nonzero.html)|Stable|N/A|

### Comparison Ops

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.allclose](https://pytorch.org/docs/2.1/generated/torch.allclose.html)|Beta|N/A|
|[torch.argsort](https://pytorch.org/docs/2.1/generated/torch.argsort.html)|Beta|N/A|
|[torch.eq](https://pytorch.org/docs/2.1/generated/torch.eq.html)|Stable|N/A|
|[torch.equal](https://pytorch.org/docs/2.1/generated/torch.equal.html)|Stable|N/A|
|[torch.ge](https://pytorch.org/docs/2.1/generated/torch.ge.html)|Beta|不支持out出参|
|[torch.greater_equal](https://pytorch.org/docs/2.1/generated/torch.greater_equal.html)|Not Support|N/A|
|[torch.gt](https://pytorch.org/docs/2.1/generated/torch.gt.html)|Stable|N/A|
|[torch.greater](https://pytorch.org/docs/2.1/generated/torch.greater.html)|Stable|N/A|
|[torch.isclose](https://pytorch.org/docs/2.1/generated/torch.isclose.html)|Stable|N/A|
|[torch.isfinite](https://pytorch.org/docs/2.1/generated/torch.isfinite.html)|Stable|N/A|
|[torch.isin](https://pytorch.org/docs/2.1/generated/torch.isin.html)|Beta|入参不支持assume_unique、invert|
|[torch.isinf](https://pytorch.org/docs/2.1/generated/torch.isinf.html)|Beta|N/A|
|[torch.isposinf](https://pytorch.org/docs/2.1/generated/torch.isposinf.html)|Not Support|N/A|
|[torch.isneginf](https://pytorch.org/docs/2.1/generated/torch.isneginf.html)|Not Support|N/A|
|[torch.isnan](https://pytorch.org/docs/2.1/generated/torch.isnan.html)|Beta|N/A|
|[torch.isreal](https://pytorch.org/docs/2.1/generated/torch.isreal.html)|Not Support|N/A|
|[torch.kthvalue](https://pytorch.org/docs/2.1/generated/torch.kthvalue.html)|Not Support|N/A|
|[torch.le](https://pytorch.org/docs/2.1/generated/torch.le.html)|Stable|N/A|
|[torch.less_equal](https://pytorch.org/docs/2.1/generated/torch.less_equal.html)|Stable|N/A|
|[torch.lt](https://pytorch.org/docs/2.1/generated/torch.lt.html)|Stable|N/A|
|[torch.less](https://pytorch.org/docs/2.1/generated/torch.less.html)|Stable|N/A|
|[torch.maximum](https://pytorch.org/docs/2.1/generated/torch.maximum.html)|Stable|N/A|
|[torch.minimum](https://pytorch.org/docs/2.1/generated/torch.minimum.html)|Stable|N/A|
|[torch.fmax](https://pytorch.org/docs/2.1/generated/torch.fmax.html)|Beta|不支持out出参|
|[torch.fmin](https://pytorch.org/docs/2.1/generated/torch.fmin.html)|Beta|不支持out出参|
|[torch.ne](https://pytorch.org/docs/2.1/generated/torch.ne.html)|Stable|N/A|
|[torch.not_equal](https://pytorch.org/docs/2.1/generated/torch.not_equal.html)|Beta|不支持out出参|
|[torch.sort](https://pytorch.org/docs/2.1/generated/torch.sort.html)|Beta|不支持out|
|[torch.topk](https://pytorch.org/docs/2.1/generated/torch.topk.html)|Beta|不支持out|
|[torch.msort](https://pytorch.org/docs/2.1/generated/torch.msort.html)|Beta|不支持out出参|

### Spectral Ops

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.stft](https://pytorch.org/docs/2.1/generated/torch.stft.html)|Beta|N/A|
|[torch.istft](https://pytorch.org/docs/2.1/generated/torch.istft.html)|Not Support|N/A|
|[torch.bartlett_window](https://pytorch.org/docs/2.1/generated/torch.bartlett_window.html)|Not Support|N/A|
|[torch.blackman_window](https://pytorch.org/docs/2.1/generated/torch.blackman_window.html)|Not Support|N/A|
|[torch.hamming_window](https://pytorch.org/docs/2.1/generated/torch.hamming_window.html)|Not Support|N/A|
|[torch.hann_window](https://pytorch.org/docs/2.1/generated/torch.hann_window.html)|Beta|不支持out、layout、device、require_grad参数|
|[torch.kaiser_window](https://pytorch.org/docs/2.1/generated/torch.kaiser_window.html)|Not Support|N/A|

### Other Operations

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.atleast_1d](https://pytorch.org/docs/2.1/generated/torch.atleast_1d.html)|Not Support|N/A|
|[torch.atleast_2d](https://pytorch.org/docs/2.1/generated/torch.atleast_2d.html)|Not Support|N/A|
|[torch.atleast_3d](https://pytorch.org/docs/2.1/generated/torch.atleast_3d.html)|Not Support|N/A|
|[torch.bincount](https://pytorch.org/docs/2.1/generated/torch.bincount.html)|Beta|不支持out出参|
|[torch.block_diag](https://pytorch.org/docs/2.1/generated/torch.block_diag.html)|Not Support|N/A|
|[torch.broadcast_tensors](https://pytorch.org/docs/2.1/generated/torch.broadcast_tensors.html)|Beta|N/A|
|[torch.broadcast_to](https://pytorch.org/docs/2.1/generated/torch.broadcast_to.html)|Stable|N/A|
|[torch.broadcast_shapes](https://pytorch.org/docs/2.1/generated/torch.broadcast_shapes.html)|Beta|N/A|
|[torch.bucketize](https://pytorch.org/docs/2.1/generated/torch.bucketize.html)|Not Support|N/A|
|[torch.cartesian_prod](https://pytorch.org/docs/2.1/generated/torch.cartesian_prod.html)|Not Support|N/A|
|[torch.cdist](https://pytorch.org/docs/2.1/generated/torch.cdist.html)|Beta|N/A|
|[torch.clone](https://pytorch.org/docs/2.1/generated/torch.clone.html)|Beta|入参不支持memory_format|
|[torch.combinations](https://pytorch.org/docs/2.1/generated/torch.combinations.html)|Not Support|N/A|
|[torch.corrcoef](https://pytorch.org/docs/2.1/generated/torch.corrcoef.html)|Not Support|N/A|
|[torch.cov](https://pytorch.org/docs/2.1/generated/torch.cov.html)|Not Support|N/A|
|[torch.cross](https://pytorch.org/docs/2.1/generated/torch.cross.html)|Not Support|N/A|
|[torch.cummax](https://pytorch.org/docs/2.1/generated/torch.cummax.html)|Not Support|N/A|
|[torch.cummin](https://pytorch.org/docs/2.1/generated/torch.cummin.html)|Not Support|N/A|
|[torch.cumprod](https://pytorch.org/docs/2.1/generated/torch.cumprod.html)|Not Support|N/A|
|[torch.cumsum](https://pytorch.org/docs/2.1/generated/torch.cumsum.html)|Stable|N/A|
|[torch.diag](https://pytorch.org/docs/2.1/generated/torch.diag.html)|Beta|入参不支持diagnoal、out|
|[torch.diag_embed](https://pytorch.org/docs/2.1/generated/torch.diag_embed.html)|Not Support|N/A|
|[torch.diagflat](https://pytorch.org/docs/2.1/generated/torch.diagflat.html)|Not Support|N/A|
|[torch.diagonal](https://pytorch.org/docs/2.1/generated/torch.diagonal.html)|Not Support|N/A|
|[torch.diff](https://pytorch.org/docs/2.1/generated/torch.diff.html)|Not Support|N/A|
|[torch.einsum](https://pytorch.org/docs/2.1/generated/torch.einsum.html)|Stable|N/A|
|[torch.flatten](https://pytorch.org/docs/2.1/generated/torch.flatten.html)|Stable|默认值start_dim为-1，torch为0|
|[torch.flip](https://pytorch.org/docs/2.1/generated/torch.flip.html)|Stable|N/A|
|[torch.fliplr](https://pytorch.org/docs/2.1/generated/torch.fliplr.html)|Not Support|N/A|
|[torch.flipud](https://pytorch.org/docs/2.1/generated/torch.flipud.html)|Not Support|N/A|
|[torch.kron](https://pytorch.org/docs/2.1/generated/torch.kron.html)|Not Support|N/A|
|[torch.rot90](https://pytorch.org/docs/2.1/generated/torch.rot90.html)|Not Support|N/A|
|[torch.gcd](https://pytorch.org/docs/2.1/generated/torch.gcd.html)|Not Support|N/A|
|[torch.histc](https://pytorch.org/docs/2.1/generated/torch.histc.html)|Stable|N/A|
|[torch.histogram](https://pytorch.org/docs/2.1/generated/torch.histogram.html)|Not Support|N/A|
|[torch.histogramdd](https://pytorch.org/docs/2.1/generated/torch.histogramdd.html)|Not Support|N/A|
|[torch.meshgrid](https://pytorch.org/docs/2.1/generated/torch.meshgrid.html)|Beta|N/A|
|[torch.lcm](https://pytorch.org/docs/2.1/generated/torch.lcm.html)|Beta|N/A|
|[torch.logcumsumexp](https://pytorch.org/docs/2.1/generated/torch.logcumsumexp.html)|Not Support|N/A|
|[torch.ravel](https://pytorch.org/docs/2.1/generated/torch.ravel.html)|Not Support|N/A|
|[torch.renorm](https://pytorch.org/docs/2.1/generated/torch.renorm.html)|Not Support|N/A|
|[torch.repeat_interleave](https://pytorch.org/docs/2.1/generated/torch.repeat_interleave.html)|Stable|N/A|
|[torch.roll](https://pytorch.org/docs/2.1/generated/torch.roll.html)|Stable|N/A|
|[torch.searchsorted](https://pytorch.org/docs/2.1/generated/torch.searchsorted.html)|Stable|N/A|
|[torch.tensordot](https://pytorch.org/docs/2.1/generated/torch.tensordot.html)|Not Support|N/A|
|[torch.trace](https://pytorch.org/docs/2.1/generated/torch.trace.html)|Stable|N/A|
|[torch.tril](https://pytorch.org/docs/2.1/generated/torch.tril.html)|Stable|N/A|
|[torch.tril_indices](https://pytorch.org/docs/2.1/generated/torch.tril_indices.html)|Not Support|N/A|
|[torch.triu](https://pytorch.org/docs/2.1/generated/torch.triu.html)|Stable|N/A|
|[torch.triu_indices](https://pytorch.org/docs/2.1/generated/torch.triu_indices.html)|Not Support|N/A|
|[torch.unflatten](https://pytorch.org/docs/2.1/generated/torch.unflatten.html)|Not Support|N/A|
|[torch.vander](https://pytorch.org/docs/2.1/generated/torch.vander.html)|Not Support|N/A|
|[torch.view_as_real](https://pytorch.org/docs/2.1/generated/torch.view_as_real.html)|Not Support|N/A|
|[torch.view_as_complex](https://pytorch.org/docs/2.1/generated/torch.view_as_complex.html)|Not Support|N/A|
|[torch.resolve_conj](https://pytorch.org/docs/2.1/generated/torch.resolve_conj.html)|Not Support|N/A|
|[torch.resolve_neg](https://pytorch.org/docs/2.1/generated/torch.resolve_neg.html)|Not Support|N/A|

### BLAS and LAPACK Operations

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.addbmm](https://pytorch.org/docs/2.1/generated/torch.addbmm.html)|Stable|N/A|
|[torch.addmm](https://pytorch.org/docs/2.1/generated/torch.addmm.html)|Beta|不支持out出参|
|[torch.addmv](https://pytorch.org/docs/2.1/generated/torch.addmv.html)|Not Support|N/A|
|[torch.addr](https://pytorch.org/docs/2.1/generated/torch.addr.html)|Not Support|N/A|
|[torch.baddbmm](https://pytorch.org/docs/2.1/generated/torch.baddbmm.html)|Stable|N/A|
|[torch.bmm](https://pytorch.org/docs/2.1/generated/torch.bmm.html)|Stable|N/A|
|[torch.chain_matmul](https://pytorch.org/docs/2.1/generated/torch.chain_matmul.html)|Not Support|N/A|
|[torch.cholesky](https://pytorch.org/docs/2.1/generated/torch.cholesky.html)|Not Support|N/A|
|[torch.cholesky_inverse](https://pytorch.org/docs/2.1/generated/torch.cholesky_inverse.html)|Not Support|N/A|
|[torch.cholesky_solve](https://pytorch.org/docs/2.1/generated/torch.cholesky_solve.html)|Not Support|N/A|
|[torch.dot](https://pytorch.org/docs/2.1/generated/torch.dot.html)|Beta|不支持out出参|
|[torch.geqrf](https://pytorch.org/docs/2.1/generated/torch.geqrf.html)|Not Support|N/A|
|[torch.ger](https://pytorch.org/docs/2.1/generated/torch.ger.html)|Not Support|N/A|
|[torch.inner](https://pytorch.org/docs/2.1/generated/torch.inner.html)|Not Support|N/A|
|[torch.inverse](https://pytorch.org/docs/2.1/generated/torch.inverse.html)|Stable|N/A|
|[torch.det](https://pytorch.org/docs/2.1/generated/torch.det.html)|Not Support|N/A|
|[torch.logdet](https://pytorch.org/docs/2.1/generated/torch.logdet.html)|Not Support|N/A|
|[torch.slogdet](https://pytorch.org/docs/2.1/generated/torch.slogdet.html)|Not Support|N/A|
|[torch.lu](https://pytorch.org/docs/2.1/generated/torch.lu.html)|Not Support|N/A|
|[torch.lu_solve](https://pytorch.org/docs/2.1/generated/torch.lu_solve.html)|Not Support|N/A|
|[torch.lu_unpack](https://pytorch.org/docs/2.1/generated/torch.lu_unpack.html)|Not Support|N/A|
|[torch.matmul](https://pytorch.org/docs/2.1/generated/torch.matmul.html)|Stable|N/A|
|[torch.matrix_power](https://pytorch.org/docs/2.1/generated/torch.matrix_power.html)|Not Support|N/A|
|[torch.matrix_exp](https://pytorch.org/docs/2.1/generated/torch.matrix_exp.html)|Not Support|N/A|
|[torch.mm](https://pytorch.org/docs/2.1/generated/torch.mm.html)|Beta|不支持out出参|
|[torch.mv](https://pytorch.org/docs/2.1/generated/torch.mv.html)|Not Support|N/A|
|[torch.orgqr](https://pytorch.org/docs/2.1/generated/torch.orgqr.html)|Not Support|N/A|
|[torch.ormqr](https://pytorch.org/docs/2.1/generated/torch.ormqr.html)|Not Support|N/A|
|[torch.outer](https://pytorch.org/docs/2.1/generated/torch.outer.html)|Stable|N/A|
|[torch.pinverse](https://pytorch.org/docs/2.1/generated/torch.pinverse.html)|Not Support|N/A|
|[torch.qr](https://pytorch.org/docs/2.1/generated/torch.qr.html)|Not Support|N/A|
|[torch.svd](https://pytorch.org/docs/2.1/generated/torch.svd.html)|Not Support|N/A|
|[torch.svd_lowrank](https://pytorch.org/docs/2.1/generated/torch.svd_lowrank.html)|Not Support|N/A|
|[torch.pca_lowrank](https://pytorch.org/docs/2.1/generated/torch.pca_lowrank.html)|Not Support|N/A|
|[torch.lobpcg](https://pytorch.org/docs/2.1/generated/torch.lobpcg.html)|Not Support|N/A|
|[torch.trapz](https://pytorch.org/docs/2.1/generated/torch.trapz.html)|Not Support|N/A|
|[torch.trapezoid](https://pytorch.org/docs/2.1/generated/torch.trapezoid.html)|Not Support|N/A|
|[torch.cumulative_trapezoid](https://pytorch.org/docs/2.1/generated/torch.cumulative_trapezoid.html)|Not Support|N/A|
|[torch.triangular_solve](https://pytorch.org/docs/2.1/generated/torch.triangular_solve.html)|Not Support|N/A|
|[torch.vdot](https://pytorch.org/docs/2.1/generated/torch.vdot.html)|Not Support|N/A|

### Foreach Operations

暂不支持

## Utilities

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[torch.compiled_with_cxx11_abi](https://pytorch.org/docs/2.1/generated/torch.compiled_with_cxx11_abi.html)|Not Support|N/A|
|[torch.result_type](https://pytorch.org/docs/2.1/generated/torch.result_type.html)|Not Support|N/A|
|[torch.can_cast](https://pytorch.org/docs/2.1/generated/torch.can_cast.html)|Not Support|N/A|
|[torch.promote_types](https://pytorch.org/docs/2.1/generated/torch.promote_types.html)|Not Support|N/A|
|[torch.use_deterministic_algorithms](https://pytorch.org/docs/2.1/generated/torch.use_deterministic_algorithms.html)|Not Support|N/A|
|[torch.are_deterministic_algorithms_enabled](https://pytorch.org/docs/2.1/generated/torch.are_deterministic_algorithms_enabled.html)|Not Support|N/A|
|[torch.is_deterministic_algorithms_warn_only_enabled](https://pytorch.org/docs/2.1/generated/torch.is_deterministic_algorithms_warn_only_enabled.html)|Not Support|N/A|
|[torch.set_deterministic_debug_mode](https://pytorch.org/docs/2.1/generated/torch.set_deterministic_debug_mode.html)|Not Support|N/A|
|[torch.get_deterministic_debug_mode](https://pytorch.org/docs/2.1/generated/torch.get_deterministic_debug_mode.html)|Not Support|N/A|
|[torch.set_float32_matmul_precision](https://pytorch.org/docs/2.1/generated/torch.set_float32_matmul_precision.html)|Not Support|N/A|
|[torch.get_float32_matmul_precision](https://pytorch.org/docs/2.1/generated/torch.get_float32_matmul_precision.html)|Not Support|N/A|
|[torch.set_warn_always](https://pytorch.org/docs/2.1/generated/torch.set_warn_always.html)|Not Support|N/A|
|[torch.get_device_module](https://pytorch.org/docs/2.1/generated/torch.get_device_module.html)|Not Support|N/A|
|[torch.is_warn_always_enabled](https://pytorch.org/docs/2.1/generated/torch.is_warn_always_enabled.html)|Not Support|N/A|
|[torch.vmap](https://pytorch.org/docs/2.1/generated/torch.vmap.html)|Not Support|N/A|
|[torch._assert](https://pytorch.org/docs/2.1/generated/torch._assert.html)|Not Support|N/A|
