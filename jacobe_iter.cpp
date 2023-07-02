#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
using namespace std;

#define epsilon 0.000001

vector<double> jacobi(vector<vector<double>> a, vector<double> b, vector<double> x_new) {
    int num = b.size();
    int n_counts = 0;
    double sum, max_precision, prev_max_precision = 1, prev_prev_max_precision = 1;
    vector<double> x_old(num);

    cout << "步骤\t解\t\t\t\t解差\t收敛阶\t收敛系数" << endl;
    do {
        n_counts++;
        max_precision = 0;
        for (int j = 0; j < num; j++) {
            x_old[j] = x_new[j];
        }

        for (int j = 0; j < num; j++) {
            sum = 0;
            for (int k = 0; k < num; k++) {
                if (j != k) {
                    sum -= a[j][k] * x_old[k];
                }
            }
            x_new[j] = (sum + b[j]) / a[j][j];
        }

        for (int i = 0; i < num; i++) {
            sum = fabs(x_new[i] - x_old[i]);
            if (sum > max_precision) {
                max_precision = sum;
            }
        }

        // 计算收敛阶
        double convergence_order = 0;
        if (fabs(prev_max_precision - prev_prev_max_precision) > epsilon && fabs(prev_max_precision) > epsilon && max_precision > epsilon) {
            convergence_order = log(max_precision / prev_max_precision) / log(prev_max_precision / prev_prev_max_precision);
        }

        // 计算收敛系数
        double convergence_factor = (prev_max_precision > epsilon) ? max_precision / prev_max_precision : 0;

        // 输出每步的计算结果
        cout << n_counts << "\t(";
        for (int i = 0; i < num; i++) {
            cout << fixed << setprecision(4) << x_new[i] << (i < num - 1 ? ", " : "");
        }
        cout << ")\t" << max_precision << "\t" << convergence_order << "\t" << convergence_factor << endl;

        prev_prev_max_precision = prev_max_precision;
        prev_max_precision = max_precision;

    } while (max_precision > epsilon);

    return x_new;
}

int main() {
    vector<vector<double>> a = {{1, 2, -2}, {1, 1, 1}, {2, 2, 1}};
    vector<double> b = {1, 2, 3};
    vector<double> x(3, 0); // 初始化向量

    x = jacobi(a, b, x);

    return 0;
}
