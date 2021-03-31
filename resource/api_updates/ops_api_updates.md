# API Updates

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.ops` operators in MindSpore, is shown in the following table.

|API|Status|Support Platform|Class
|:----|:----|:--------------|:-------
|mindspore.ops.Dihedral14LJEnergy|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.PMEExcludedForce|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.AngleAtomEnergy|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.DihedralForce|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.AngleForceWithAtomEnergy|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.Dihedral14LJForce|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.BondForceWithAtomVirial|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.DihedralAtomEnergy|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.Dihedral14CFAtomEnergy|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.LJForceWithPMEDirectForce|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.Dihedral14CFEnergy|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.AngleEnergy|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.BondEnergy|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.MDIterationLeapFrog|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.NeighborListUpdate|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.DihedralEnergy|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.Dihedral14LJAtomEnergy|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.PMEEnergy|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.LJEnergy|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.BondForce|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.DihedralForceWithAtomEnergy|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.PMEReciprocalForce|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.BondAtomEnergy|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.Dihedral14LJForceWithDirectCF|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.LJForce|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.Dihedral14LJCFForceWithAtomEnergy|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.BondForceWithAtomEnergy|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.AngleForce|New|r1.2: GPU|operators--Sponge Operators
|mindspore.ops.Dropout2D|New|r1.2: Ascend|operators--Neural Network Operators
|mindspore.ops.BCEWithLogitsLoss|New|r1.2: Ascend|operators--Neural Network Operators
|mindspore.ops.Dropout3D|New|r1.2: Ascend|operators--Neural Network Operators
|mindspore.ops.Conv3D|New|r1.2: Ascend|operators--Neural Network Operators
|mindspore.ops.Conv3DTranspose|New|r1.2: Ascend|operators--Neural Network Operators
|mindspore.ops.SeLU|New|r1.2: Ascend|operators--Neural Network Operators
|mindspore.ops.Mish|New|r1.2: Ascend|operators--Neural Network Operators
|mindspore.ops.NLLLoss|New|r1.2: Ascend|operators--Neural Network Operators
|mindspore.ops.MaxPool3D|New|r1.2: Ascend|operators--Neural Network Operators
|mindspore.ops.MulNoNan|New|r1.2: Ascend|operators--Math Operators
|mindspore.ops.Randperm|New|r1.2: Ascend|operators--Math Operators
|mindspore.ops.NoRepeatNGram|New|r1.2: Ascend|operators--Math Operators
|mindspore.ops.stack|New|r1.2: same as mindspore.ops.Stack|functional
|mindspore.ops.add|New|r1.2: same as mindspore.ops.Add|functional
|mindspore.ops.batch_dot|New|r1.2: Ascend/GPU/CPU|composite
|mindspore.ops.matmul|New|r1.2: Ascend/GPU/CPU|composite
|mindspore.ops.dot|New|r1.2: Ascend/GPU/CPU|composite
|mindspore.ops.MakeRefKey|Deleted|r1.1: Ascend/GPU/CPU|operators--Other Operators
|mindspore.ops.Unpack|Deleted|r1.1: To Be Developed|operators--Neural Network Operators
|mindspore.ops.FastGelu|Deleted|r1.1: To Be Developed|operators--Neural Network Operators
|mindspore.ops.FusedBatchNorm|Deleted|r1.1: CPU|operators--Neural Network Operators
|mindspore.ops.Pack|Deleted|r1.1: To Be Developed|operators--Neural Network Operators
|mindspore.ops.FusedBatchNormEx|Deleted|r1.1: GPU|operators--Neural Network Operators
|mindspore.ops.Gelu|Deleted|r1.1: To Be Developed|operators--Neural Network Operators
|mindspore.ops.TensorAdd|Deleted|r1.1: To Be Developed|operators--Math Operators
|mindspore.ops.GatherV2|Deleted|r1.1: To Be Developed|operators--Array Operators
|mindspore.ops.control_depend|Deleted|r1.1: same as mindspore.ops.ControlDepend|functional
|mindspore.ops.tensor_add|Deleted|r1.1: same as mindspore.ops.Add|functional
|mindspore.ops.pack|Deleted|r1.1: same as mindspore.ops.Pack|functional
|mindspore.ops.ControlDepend|Deleted|r1.1: Ascend/GPU/CPU|Control Flows
|mindspore.ops.add_flags|Deleted|r1.1: To Be Developed|composite
|mindspore.ops.Acosh| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.Tanh| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.SmoothL1Loss| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.MirrorPad| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.ApplyCenteredRMSProp| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.Pad| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.BinaryCrossEntropy| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.TopK| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.GeLU| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.CTCGreedyDecoder| Changed |r1.1: To Be Developed => r1.2: Ascend|operators--Neural Network Operators
|mindspore.ops.HSigmoid| Changed |r1.1: GPU => r1.2: GPU/CPU|operators--Neural Network Operators
|mindspore.ops.L2Normalize| Changed |r1.1: Ascend => r1.2: Ascend/GPU|operators--Neural Network Operators
|mindspore.ops.ResizeBilinear| Changed |r1.1: Ascend => r1.2: Ascend/CPU|operators--Neural Network Operators
|mindspore.ops.Stack| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.Elu| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.Unstack| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.Sigmoid| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.LayerNorm| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.LogSoftmax| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.Conv2DBackpropInput| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.HSwish| Changed |r1.1: GPU => r1.2: GPU/CPU|operators--Neural Network Operators
|mindspore.ops.ApplyRMSProp| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.CTCLoss| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.BatchNorm| Changed |r1.1: Ascend => r1.2: Ascend/CPU|operators--Neural Network Operators
|mindspore.ops.Adam| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Neural Network Operators
|mindspore.ops.Minimum| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.LogicalNot| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.Sin| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.Floor| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.Mod| Changed |r1.1: Ascend => r1.2: Ascend/CPU|operators--Math Operators
|mindspore.ops.SquareSumAll| Changed |r1.1: Ascend => r1.2: Ascend/GPU|operators--Math Operators
|mindspore.ops.Asin| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.Asinh| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.Tan| Changed |r1.1: Ascend => r1.2: Ascend/CPU|operators--Math Operators
|mindspore.ops.Reciprocal| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.Greater| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.Expm1| Changed |r1.1: Ascend => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.CumSum| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.LogicalOr| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.Div| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.ReduceAny| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.BatchMatMul| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.Sinh| Changed |r1.1: Ascend => r1.2: Ascend/CPU|operators--Math Operators
|mindspore.ops.Atan2| Changed |r1.1: Ascend => r1.2: Ascend/CPU|operators--Math Operators
|mindspore.ops.ACos| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.Cosh| Changed |r1.1: Ascend => r1.2: Ascend/CPU|operators--Math Operators
|mindspore.ops.FloorDiv| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.Cos| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.Atan| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.Erfc| Changed |r1.1: Ascend => r1.2: Ascend/GPU|operators--Math Operators
|mindspore.ops.Atanh| Changed |r1.1: Ascend => r1.2: Ascend/CPU|operators--Math Operators
|mindspore.ops.Log1p| Changed |r1.1: Ascend => r1.2: Ascend/GPU|operators--Math Operators
|mindspore.ops.LogicalAnd| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.ReduceAll| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.SquaredDifference| Changed |r1.1: Ascend => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.GreaterEqual| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Math Operators
|mindspore.ops.Print| Changed |r1.1: Ascend => r1.2: Ascend/GPU|operators--Debug Operators
|mindspore.ops.HyperMap| Changed |r1.1: To Be Developed => r1.2: Ascend/GPU/CPU|composite
|mindspore.ops.gamma| Changed |r1.1: Ascend/GPU/CPU => r1.2: Ascend|composite
|mindspore.ops.MultitypeFuncGraph| Changed |r1.1: To Be Developed => r1.2: Ascend/GPU/CPU|composite
|mindspore.ops.count_nonzero| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|composite
|mindspore.ops.poisson| Changed |r1.1: Ascend/GPU/CPU => r1.2: Ascend|composite
|mindspore.ops.GradOperation| Changed |r1.1: To Be Developed => r1.2: Ascend/GPU/CPU|composite
|mindspore.ops.Broadcast| Changed |r1.1: , /GPU => r1.2: Ascend/GPU|operators--Common Operators
|mindspore.ops.TensorScatterUpdate| Changed |r1.1: Ascend => r1.2: Ascend/GPU|operators--Array Operators
|mindspore.ops.Squeeze| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Array Operators
|mindspore.ops.GatherNd| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Array Operators
|mindspore.ops.Gather| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Array Operators
|mindspore.ops.ArgMinWithValue| Changed |r1.1: Ascend => r1.2: Ascend/CPU|operators--Array Operators
|mindspore.ops.GatherD| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Array Operators
|mindspore.ops.UnsortedSegmentSum| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Array Operators
|mindspore.ops.ResizeNearestNeighbor| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|operators--Array Operators

>
