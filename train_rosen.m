function [WsSA, WsAR, Ts] = train_rosen(train_X, train_y, val_X, val_y)
	function [mean_, std_] = mean_std(X)
		mean_ = mean(X(:));
		std_  = std(X(:));
	end
	batch_size = 40;
	epochs = 10;
	batch_count = idivide(length(train_y), batch_size);

	dropout = 0.20;
	weight_smoothing = 0.1 ^ (1 / batch_count);

	S_count = size(train_X, 2);
	A_count = 100;
	R_count = 10;

	WsSA = (randi(3, A_count, S_count) - 2);
	WsAR = zeros(R_count, A_count);
	WsAR_smooth = WsAR;

	T_mean = 0;
	T_std = std((WsSA * train_X(1:200,:)')(:));
	Ts = round(0.1 * (randn(A_count, 1)) * T_std);

	train_accs = [];
	val_accs = [];

	for epoch = 1:epochs
		begin_clock = clock();
		perm = randperm(length(train_y));
		batch_correct_count = 0;
		cur_X = train_X(perm,:);
		cur_y = train_y(perm,:);

		for batchn = 1:batch_count
			batch_X = cur_X((1+(batchn-1)*batch_size):batchn*batch_size,:);
			batch_y = cur_y((1+(batchn-1)*batch_size):batchn*batch_size);

			A_map = (WsSA * batch_X' > Ts);
			A_map = A_map .* (rand(size(A_map)) >= dropout);
			R_map = (WsAR * A_map)';
			R_map_correct = bsxfun(@eq, batch_y, 0:(R_count-1));

			[~, ind_] = max(R_map, [], 2);
			batch_correct_count = batch_correct_count + sum((ind_ - 1) == batch_y);

			WsAR = WsAR + (A_map * ((R_map > 0 != R_map_correct) .* (2 * R_map_correct - 1)))';
			WsAR_smooth = weight_smoothing * WsAR_smooth +(1 - weight_smoothing) * (WsAR / norm(WsAR));
			printf('Epoch %d %.2f%%: train acc %.2f%% \r', epoch, 100 * batchn / batch_count, 100 * batch_correct_count / (batchn * batch_size));
		end

		time_diff = etime(clock(), begin_clock);
		train_acc = batch_correct_count / (batch_count * batch_size);
		val_acc = mean(infer_rosen(val_X, WsSA, WsAR_smooth, Ts) == val_y);
		printf('Epoch %d: train acc %.2f%%, val acc %.2f%% (%.1fs)\n',
			epoch,
			100 * train_acc,
			100 * val_acc,
			time_diff);

		train_accs = [train_accs train_acc];
		val_accs   = [val_accs   val_acc];
	end

	WsAR = WsAR_smooth;

	%hold on;
	%grid on;
	%plot(train_accs);
	%plot(val_accs);
	%xlabel('Эпоха');
	%ylabel('Точность распознавания');
	%hold off;
end
