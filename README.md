# rosenblatt
Simple rosenblatt perceptron that trains on MNIST.

Usage:
1. Create `mnist/` directory and put ungz'ed data in it.
2. Run following:
	```
	load_mnist;
	[WsSA, WsAR, Ts] = train_rosen(train_X, train_y, test_X, test_y);
	pred = infer_rosen(test_X, WsSA, WsAR, Ts);
	```
