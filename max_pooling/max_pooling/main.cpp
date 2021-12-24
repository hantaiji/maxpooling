//#pragma GCC optimize(s)
#include "temp.h"

int main()
{
	default_random_engine dre;
	const int lower = 0;
	const int upper = 255;
	uniform_int_distribution<int> random_pixel(lower, upper);
	int batch,channel,row, col;
	cout << "enter the row and col of matrix: " << endl;
	cin >> batch >> channel >> row >> col;
	vector<vector<vector<vector<int>>>>  matrix(batch, vector<vector<vector<int>>>(channel, vector< vector<int>>(row, vector<int>(col,-1))));//初始化维度大小为n*m*x*y，初值为z。
	vector<vector<vector<vector<int>>>>  src2(32, vector<vector<vector<int>>>(1, vector< vector<int>>(56, vector<int>(56, -1))));
	vector<vector<vector<vector<int>>>>  result;//初始化维度大小为n*m*x*y，初值为z。
	vector<vector<vector<vector<int>>>>  addResult(32, vector<vector<vector<int>>>(64, vector< vector<int>>(56, vector<int>(56, -1))));

#pragma region 生成四维矩阵
	clock_t t1 = clock();
    #pragma omp parallel for
	for (int m = 0; m < batch; m++)
		for (int n = 0; n < channel; n++)
			for (int i = 0; i < row; ++i)
				for (int j = 0; j < col; ++j)
					matrix[m][n][i][j] = random_pixel(dre);
	#pragma omp parallel for
	for (int m = 0; m < 32; m++)
			for (int i = 0; i < 56; ++i)
				for (int j = 0; j < 56; ++j)
					src2[m][0][i][j] = random_pixel(dre);
	clock_t t2 = clock();
	double totaltime = (t2 - t1)*1.0 / CLOCKS_PER_SEC;
	cout << "矩阵生成时间:" << totaltime << "秒" << endl;
	cout << "=====================" << endl;
	//矩阵展示
	//cout << "the matrix: " << endl;
	//for (int i = 0; i < batch; i++)
	//	for (int j = 0; j < channel; j++)
	//	{
	//		for (const vector<int> &row_val : matrix[i][j])
	//		{
	//			for (const int &val : row_val)
	//				cout << left << setw(4) << val << " ";
	//			cout << endl;
	//		}
	//	}
#pragma endregion
	//池化
	cout << "=====================" << endl;
	maxPooling myMaxPooling(matrix, batch, channel);

#pragma region 输出池化结果
	//for (int i = 0; i < batch; i++)
	//	for (int j = 0; j < channel; j++)
	//	{
	//		for (const vector<int> &row_val : myMaxPooling.maxPoolingResult[i][j])
	//		{
	//			for (const int &val : row_val)
	//				cout << left << setw(4) << val << " ";
	//			cout << endl;
	//		}
	//	}
	//cout << "=====================" << endl;
	//for (int i = 0; i < batch; i++)
	//	for (int j = 0; j < channel; j++)
	//	{
	//		for (const vector<int> &row_val : myMaxPooling.maxPoolingResult_SW[i][j])
	//		{
	//			for (const int &val : row_val)
	//				cout << left << setw(4) << val << " ";
	//			cout << endl;
	//		}
	//	}
#pragma endregion
	//四维矩阵相加
	#pragma omp parallel for
	for (int m = 0; m < 32; m++)
		for (int n = 0; n < 64; n++)
			for (int i = 0; i < 56; ++i)
				for (int j = 0; j < 56; ++j)
					addResult[m][n][i][j] = matrix[m][n][i][j] + src2[m][0][i][j];
	system("pause");
	return 0;
}