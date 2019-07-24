#include "heap.h"

using namespace std;

void top_k(vector<point>& input_arr, int32_t n, int32_t k) {
    // O(k)
    // we suppose the k element of the min heap if the default top k element
    min_heap_t min_heap(input_arr, k);
    min_heap.build_heap_from_bottom_to_top();
    
    for (int32_t i = k; i < n; ++i) {
        // compare each element with the min element of the min heap
        // if the element > the min element of the min heap
        // we think may be the element is one of what we wanna to find in the top k
        if (input_arr[i].semi > min_heap.arr[0].semi){
            // swap
            min_heap.arr[0] = input_arr[i];
            
            // heap adjust
            min_heap.heap_adjust_from_top_to_bottom(0, k - 1);
        }
    }
    
    input_arr.assign(min_heap.arr.begin(),min_heap.arr.end());
}

void bottom_k(vector<point>& input_arr, int32_t n, int32_t k) {
    // O(k)
    // we suppose the k element of the max heap if the default top k element
    max_heap_t max_heap(input_arr, k);
    max_heap.build_heap_from_bottom_to_top();
    
    for (int32_t i = k; i < n; ++i) {
        // compare each element with the max element of the max heap
        // if the element < the max element of the max heap
        // we think may be the element is one of what we wanna to find in the top k
        if (input_arr[i].semi < max_heap.arr[0].semi){
            // swap
            max_heap.arr[0] = input_arr[i];
            
            // heap adjust
            max_heap.heap_adjust_from_top_to_bottom(0, k - 1);
        }
    }
    
    input_arr.assign(max_heap.arr.begin(),max_heap.arr.end());
}