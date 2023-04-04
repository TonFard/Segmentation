 * 项目结构
	* model下面存各种分割的模型
	* train.py用于训练
	* predict.py用于训练之后结果的预测
	* evaluate.py用于各种指标的计算

* 注意事项
	* 模型修改在train.py中的create_model中修改model即可
	* 训练前注意数据集的格式 
	* > root_path
	* >> train
	* >>> image
	* >>> mask

	* >> val
	* >>> image
	* >>> mask 
	* mask中图片的像素要对应分割的类别数目，0默认为背景对应的类别，1，2，3等等为相应的目标区域对应的像素
	* 使用某些模型进行分割是图片的大小需要保持一致，例如Unet++，可以通过pad_size.py对图片大小进行修改
