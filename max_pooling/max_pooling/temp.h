#pragma once
#include <vector>
#include <iostream>
#include <random>
#include <deque>
#include <queue>
#include <iomanip>
#include <ctime>
#include <omp.h>

using namespace std;

class maxPooling
{
public:
	vector<vector<vector<vector<int>>>> maxPoolingResult, maxPoolingResult_SW;
	vector<vector<int>> poolingTemp;
	maxPooling(vector<vector<vector<vector<int>>>>& src, int& batch, int&channel);
	vector<vector<int>> padding(vector<vector<int>>& src, vector<int> padsize = { 1,1 });
	vector<vector<int>> pooling(vector<vector<int>>& matrix, vector<int> kernel = { 3,3 }, vector<int> stride = { 2,2 });
	vector<vector<int>> pooling2(vector<vector<int>>& matrix, vector<int> kernel = { 3,3 }, vector<int> stride = { 2,2 });
	//�ݼ����л�������ʵ�ֳػ�
	vector<int> maxSlidingWindow(vector<int>& nums, int k, int stride = 2);//��������Ѱ�����ֵ
	vector<vector<int>> transpose(vector<vector<int> >& matrix);
	vector<vector<int>> poolingSW(vector<vector<int> >& matrix, int k_row, int k_col);
	vector<vector<int>> poolingSW2(vector<vector<int> >& matrix, int k_row, int k_col);
	//��ά�����������
	void showMatrix(vector<vector<int>> &matrix);
private:
	vector<int> _padsize = { 1,1 }, _kernel = { 3,3 }, _stride = { 2,2 };
};
