import time 
import numpy as np

###########################################################################################
"""
    Sorting Algorithm Generator Functions end here
"""

def insertion_sort(arr):
    """
        para: arr: list of integers
        
        description: This function sorts the given list of integers using the insertion sort algorithm.
        
        return: sorted list of integers
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def insertion_sort_yield(arr):
    """
    Generator function for insertion sort to yield array after each step.
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        yield arr  

def selection_sort(arr):
    """
    para: list of integers

    description: This function sorts the given list of integers using the selection sort algorithm.

    returns: sorted list of integers
    """
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        yield arr  # Yielding after each swap

def bubble_sort(arr):
    """
        para: arr: list of integers
        
        description: This function sorts the given list of integers using the bubble sort algorithm.
        
        return: sorted list of integers
    """
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
        yield arr  # Yielding after each outer loop completion

def merge_sort(arr):
    """
    para: arr: list of integers
    
    description: This function sorts the given list of integers using the merge sort algorithm.
    
    return: sorted list of integers
    """
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]
        
        merge_sort(left_half)
        merge_sort(right_half)
        
        i = j = k = 0
        
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1
        
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

    yield arr  # Yielding the final sorted array

def quick_sort(arr):
    """
    para: arr: list of integers
    
    description: This function sorts the given list of integers using the quick sort algorithm and yields
    the array at each significant step.
    
    yield: array after each partition step
    """
    if len(arr) <= 1:
        yield arr  
    else:
        pivot = arr[0]
        less = [x for x in arr[1:] if x <= pivot]
        greater = [x for x in arr[1:] if x > pivot]

        sorted_less = list(quick_sort(less))
        sorted_greater = list(quick_sort(greater))

        result = sorted_less + [pivot] + sorted_greater
        yield result  

def heapify(arr, n, i):
    """
    para: arr: list of integers
          n: integer
          i: integer
          
    description: This function heapifies the given list of integers.
    
    yield: None
    """
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[i] < arr[left]:
        largest = left

    if right < n and arr[largest] < arr[right]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    """
    para: arr: list of integers
    
    description: This function sorts the given list of integers using the heap sort algorithm.
    
    yield: sorted list of integers
    """
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

    yield arr

def counting_sort(arr):
    """
    para: arr: list of integers
    
    description: This function sorts the given list of integers using the counting sort algorithm.
    
    yield: sorted list of integers
    """
    max_val = max(arr)
    
    count = [0] * (max_val + 1)
    
    for num in arr:
        count[num] += 1
    
    for i in range(1, len(count)):
        count[i] += count[i - 1]
    
    output = [0] * len(arr)
    
    for num in reversed(arr):
        output[count[num] - 1] = num
        count[num] -= 1
    
    yield output

def counting_sort_for_radix(arr, exp):
    """
    para: arr: list of integers
            exp: integer, represent the current digit's place value (units, tens, hundreds, etc.)
            
    description: This function sorts the given list of integers using the counting sort algorithm for radix sort.
    
    yield: None
    """
    n = len(arr)
    
    output = [0] * n
    
    count = [0] * 10
    
    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1
    
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    i = n - 1
    while i >= 0:
        index = arr[i] // exp 
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1
    
    for i in range(len(arr)):
        arr[i] = output[i]

def radix_sort(arr):
    """
    para: arr: list of integers
    
    description: This function sorts the given list of integers using the radix sort algorithm.
    
    yield: sorted list of integers
    """
    max_val = max(arr)
    
    exp = 1
    while max_val // exp > 0:
        counting_sort_for_radix(arr, exp)
        exp *= 10
    
    yield arr

def bucket_sort(arr):
    """
    para: arr: list of integers
    
    description: This function sorts the given list of integers using the bucket sort algorithm.
    
    yield: sorted list of integers
    """
    if len(arr) == 0:
        yield arr

    bucket_count = len(arr)
    max_value = max(arr)
    buckets = [[] for _ in range(bucket_count)]
    
    for num in arr:
        index = int(num * bucket_count / (max_value + 1))
        buckets[index].append(num)

    for bucket in buckets:
        bucket.sort()

    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(bucket)

    yield sorted_arr

###########################################################################################
"""
    Sorting Algorithm Generator Functions end here
"""
###########################################################################################
"""
    Streamlit Functions start here
"""
import streamlit as st
import plotly.graph_objects as go


time_complexities = {
    "Insertion Sort": "O(n^2)",
    "Selection Sort": "O(n^2)",
    "Bubble Sort": "O(n^2)",
    "Merge Sort": "O(n log n)",
    "Quick Sort": "O(n log n)",
    "Heap Sort": "O(n log n)",
    "Counting Sort": "O(n)",
    "Radix Sort": "O(n)",
    "Bucket Sort": "O(n^2) (worst case)",
}

def display_time_complexity_graph():
    fig = go.Figure()

    algorithms = list(time_complexities.keys())
    complexities = [2, 2, 2, np.log2(10), np.log2(10), np.log2(10), 1, 1, 2]  # Dummy values for representation

    colors = ['red' if 'n^2' in tc else 'orange' if 'n log n' in tc else 'green' for tc in time_complexities.values()]

    fig.add_trace(go.Bar(x=algorithms, y=complexities, marker_color=colors))
    
    fig.update_layout(
        title="Time Complexity of Sorting Algorithms",
        xaxis_title="Sorting Algorithms",
        yaxis_title="Complexity (Dummy Scale)",
        yaxis=dict(tickvals=[0, 1, 2], ticktext=["O(1)", "O(n)", "O(n^2)"]),
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0, 0, 0, 0)'  # Transparent background
    )

    return fig

def real_time_sorting_visualization(array, sorting_gen, algorithm, selected_speed):
    """
    Function to visualize the sorting process in real-time using Streamlit and Plotly.

    Parameters:
    - array: The initial array to be sorted.
    - sorting_gen: The generator function that yields the array at each step.
    - algorithm: The name of the sorting algorithm.
    - selected_speed: The speed of visualization in seconds.
    """
    plot_placeholder = st.empty()  # Placeholder for the plot

    iteration_count = 0
    
    for current_array in sorting_gen(array):
        iteration_count += 1
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(x=list(range(len(current_array))), y=current_array, marker_color='blue'))

        fig.update_layout(title=f"{algorithm} Visualization - Step {iteration_count}",
                          xaxis_title="Index",
                          yaxis_title="Value",
                          xaxis=dict(tickmode='linear'),
                          yaxis=dict(range=[0, max(current_array) + 10]), # Keep a consistent y-axis
                          plot_bgcolor='rgba(0, 0, 0, 0)') # Transparent background

        plot_placeholder.plotly_chart(fig, use_container_width=True)
        
        time.sleep(selected_speed)
        
      
def reset():
    st.session_state.selection = 'Please Select'

###########################################################################################
"""
    Streamlit Functions end here
"""  