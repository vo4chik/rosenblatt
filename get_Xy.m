function [X, y] = get_Xy(images_name, labels_name)
	fimages = fopen(images_name);
	himages = swapbytes(fread(fimages, 4, '*uint32'));
	images_data = fread(fimages, 'uint8');
	X = permute(reshape(images_data, himages(3) * himages(4), himages(2)), [2 1 3]);
	fclose(fimages);

	flabels = fopen(labels_name);
	hlabels = swapbytes(fread(flabels, 2, '*uint32'));
	labels_data = fread(flabels, 'uint8');
	y = labels_data;
	fclose(flabels);
end
