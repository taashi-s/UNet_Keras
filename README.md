# traial UNet (with Keras)

## Current problem
```
InvalidArgumentError (see above for traceback): Incompatible shapes: [1635920] vs. [2000]
         [[Node: metrics/dice_coef/mul = Mul[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:GPU:0"](metrics/dice_coef/Reshape, metrics/dice_coef/Reshape_1)]]
         [[Node: metrics/dice_coef/Mean/_337 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_2268_metrics/dice_coef/Mean", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]
```
