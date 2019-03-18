function res = infer_rosen(X, WsSA, WsAR, Ts)
	A_map = (WsSA * X' > Ts);
	R_map = (WsAR * A_map)';
	[val, ind] = max(R_map, [], 2);
	res = (ind - 1);
end
