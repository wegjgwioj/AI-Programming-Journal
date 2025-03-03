#include <vector>
#include <unordered_map>
#include <iostream>
using namespace std;

class Solution {
public:
    int numIdenticalPairs(vector<int>& nums) {
        unordered_map<int, int> countMap; // 用于统计每个数字的出现次数
        int result = 0; // 初始化结果为0

        // 统计每个数字的出现次数
        for (int num : nums) {
            countMap[num]++;
        }

        // 计算好数对的数量
        for (auto& [num, cnt] : countMap) {
            if (cnt >= 2) {
                result += cnt * (cnt - 1) / 2; // 组合数公式
            }
        }

        return result; // 返回最终结果
    }
};