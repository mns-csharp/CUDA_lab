#include <iostream>
#include <vector>

std::vector<int> get_factors(int x) {
    std::vector<int> factors;
    int largest_factor = x;
    while (largest_factor % 2 == 0 && largest_factor > 32) {
        largest_factor /= 2;
    }
    factors.push_back(largest_factor);
    int remaining_factor = x / largest_factor;
    int next_factor = remaining_factor;
    while (next_factor % 2 == 0 && next_factor > 8) {
        next_factor /= 2;
    }
    factors.push_back(next_factor);
    factors.push_back(remaining_factor / next_factor);
    return factors;
}

int main() {
    int x = 512;
    std::vector<int> factors = get_factors(x);
    for (int factor : factors) {
        std::cout << factor << " ";
    }
    std::cout << std::endl;
    return 0;
}
