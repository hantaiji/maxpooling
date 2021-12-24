#include "temp.h"

maxPooling::maxPooling(vector<vector<vector<vector<int>>>>& src, int& batch, int&channel)
{
#pragma region 循环方法
	cout << "循环方法:" << endl;
	clock_t t1 = clock();
	//vector<int> k = { 7,7 };
	for (int i = 0; i < batch; i++)
	{
		vector<vector<vector<int>>> matrixTemp;
		for (int j = 0; j < channel; j++)
		{
			vector<vector<int>> matrixAfterPad = padding(src[i][j]);
			//cout << "the matrix after padding: " << endl;
			//showMatrix(matrixAfterPad);
			this->poolingTemp = pooling2(matrixAfterPad);
			//cout << "=====================" << endl;
			//cout << "the matrix after pooling: " << endl;
			//showMatrix(this->poolingTemp);
			matrixTemp.push_back(this->poolingTemp);
		}
		this->maxPoolingResult.push_back(matrixTemp);
	}
	clock_t t2 = clock();
	double totaltime = (t2 - t1)*1.0 / CLOCKS_PER_SEC;
	cout << "循环方法执行时间:" << totaltime << "秒" << endl;
#pragma endregion
	cout << "=====================" << endl;
//#pragma region 滑动窗口方法
//	cout << "滑动窗口方法:" << endl;
//	clock_t t3 = clock();
//	for (int i = 0; i < batch; i++)
//	{
//		vector<vector<vector<int>>> matrixTemp;
//		//vector<vector<vector<int>>> matrixTemp2;
//		for (int j = 0; j < channel; j++)
//		{
//			vector<vector<int>> matrixAfterPad = padding(src[i][j]);
//			//cout << "the matrix after padding: " << endl;
//			//showMatrix(matrixAfterPad);
//			vector<vector<int>> poolingResult_SW = poolingSW2(matrixAfterPad, 3, 3);
//			//vector<vector<int>> poolingResult_SW2 = poolingSW2(matrixAfterPad, 3, 3);
//			//cout << "=====================" << endl;
//			//cout << "the matrix after poolingSW: " << endl;		
//			//showMatrix(poolingResult_SW);
//		/*	cout << "=====================" << endl;
//			cout << "the matrix after poolingSW2: " << endl;
//			showMatrix(poolingResult_SW2);*/
//			matrixTemp.push_back(poolingResult_SW);
//		}
//		this->maxPoolingResult_SW.push_back(matrixTemp);
//	}
//	clock_t t4 = clock();
//	totaltime = (t4 - t3)*1.0 / CLOCKS_PER_SEC;
//	cout << "滑动窗口执行时间:" << totaltime << "秒" << endl;
//#pragma endregion
}
vector<vector<int>> maxPooling::padding(vector<vector<int>>&src, vector<int> padsize)
{
	int height = src.size();
	int width = src[0].size();
	vector<vector<int> > matrix(height + 2 * padsize[0], vector<int>(width + 2 * padsize[1], 0));
	#pragma omp parallel for
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			matrix[i + padsize[0]][j + padsize[1]] = src[i][j];
		}
	}
	return matrix;
}

vector<vector<int>> maxPooling::pooling(vector<vector<int>>& matrix, vector<int> kernel, vector<int> stride)
{
	int row = matrix.size(), col = matrix[0].size();
	int pooledRow = (row - kernel[0]) / stride[1] + 1, pooledCol = (col - kernel[1]) / stride[0] + 1;
	vector<vector<int>> pooledMatrix;
	//vector<vector<int>> pooledMatrix(row / 2 - 1, vector<int>(col / 2 - 1, -1));
	//#pragma omp parallel for
	for (int i = 0; i <= col - kernel[1]; i=i+2)
	{
		vector<int> rowResult;
		for (int j = 0; j <= row - kernel[0]; j=j+2)
		{
			int max = matrix[i][j];
			for(int m = 0; m < kernel[0]; m++)
				for (int n = 0; n < kernel[1]; n++)
				{
					if (matrix[i+m][j+n] >= max)
						max = matrix[i + m][j + n];
				}
			rowResult.push_back(max);
			//pooledMatrix[i / 2][j / 2] = max;
		}
		pooledMatrix.push_back(rowResult);
	}
	return pooledMatrix;
}

vector<vector<int>> maxPooling::pooling2(vector<vector<int>>& matrix, vector<int> kernel, vector<int> stride)
{
	int row = matrix.size(), col = matrix[0].size();
	int pooledRow = (row - kernel[0]) / stride[1] + 1, pooledCol = (col - kernel[1]) / stride[0] + 1;
	//vector<vector<int>> pooledMatrix;
	vector<vector<int>> pooledMatrix(row / 2 - 1, vector<int>(col / 2 - 1, -1));
	#pragma omp parallel for
	for (int i = 0; i <= col - kernel[1]; i = i + 2)
	{
		//vector<int> rowResult;
		for (int j = 0; j <= row - kernel[0]; j = j + 2)
		{
			int max = matrix[i][j];
			for (int m = 0; m < kernel[0]; m++)
				for (int n = 0; n < kernel[1]; n++)
				{
					if (matrix[i + m][j + n] >= max)
						max = matrix[i + m][j + n];
				}
			//rowResult.push_back(max);
			pooledMatrix[i / 2][j / 2] = max;
		}
		//pooledMatrix.push_back(rowResult);
	}
	return pooledMatrix;
}

void maxPooling::showMatrix(vector<vector<int>> &matrix)
{
	int matrix_h = matrix.size(), matrix_w = matrix[0].size();
    //#pragma omp parallel for//循环输出不可使用openMP
	for (int i = 0; i < matrix_h; i++) {
		for (int j = 0; j < matrix_w; j++) {
			cout << left << setw(4) << matrix[i][j] << " ";
		}
		cout << endl;
	}
}

vector<int> maxPooling::maxSlidingWindow(vector<int>& nums, int k, int stride)
{
	int n = nums.size();
	deque<int> q;
	//#pragma omp parallel for
	for (int i = 0; i < k; i++) {
		while (!q.empty() && nums[i] >= nums[q.back()]) {
			q.pop_back();
		}
		q.push_back(i);
	}

	vector<int> ans = { nums[q.front()] };
	for (int i = k; i < n; i++) {
		while (!q.empty() && nums[i] >= nums[q.back()]) {
			q.pop_back();
		}
		q.push_back(i);
		while (q.front() <= i - k) {
			q.pop_front();
		}
		if (i%stride == 0)//每2次执行一次push_back，即步长为2
			ans.push_back(nums[q.front()]);
	}
	return ans;
}

vector<vector<int>> maxPooling::transpose(vector<vector<int> >& matrix)
{
	if (matrix.empty()) return vector<vector<int>>();
	int n = matrix.size(), m = matrix[0].size();
	vector<vector<int>> transposed(m, vector<int>(n, -1));
	#pragma omp parallel for
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			transposed[j][i] = matrix[i][j];
	return transposed;
}

vector<vector<int>> maxPooling::poolingSW(vector<vector<int> >& matrix, int k_row, int k_col)
{
	vector<vector<int>> row_result;
	for (vector<int>& row_vals : matrix)
		row_result.push_back(maxSlidingWindow(row_vals, k_col));
	vector<vector<int> > row_result_transposed = transpose(row_result);
	vector<vector<int> > result_transposed;
	for (vector<int>& col_vals : row_result_transposed)
		result_transposed.push_back(maxSlidingWindow(col_vals, k_row));
	vector<vector<int> > result = transpose(result_transposed);
	return result;
}

vector<vector<int>> maxPooling::poolingSW2(vector<vector<int> >& matrix, int k_row, int k_col)
{
	
	int row = matrix.size();
	vector<vector<int>> row_result(row);//需要声明第一维，否则vector out range
	//for (vector<int>& row_vals : matrix)
		//row_result.push_back(maxSlidingWindow(row_vals, k_col));
	#pragma omp parallel for
	for (int i = 0; i < row; i++)
	{
			row_result[i] = maxSlidingWindow(matrix[i], k_col);
	}
	vector<vector<int> > row_result_transposed = transpose(row_result);
	
	int col = row_result_transposed.size();
	vector<vector<int> > result_transposed(col);//需要声明第一维，否则vector out range
	//for (vector<int>& col_vals : row_result_transposed)
		//result_transposed.push_back(maxSlidingWindow(col_vals, k_row));
	#pragma omp parallel for
	for (int i = 0; i < col; i++)
	{
		result_transposed[i] = maxSlidingWindow(row_result_transposed[i], k_col);
	}
	vector<vector<int> > result = transpose(result_transposed);
	return result;
}