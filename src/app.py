import time
import numpy as np
from function import *
import pyautogui
import plotly.graph_objects as go
import streamlit as st  

speed_mapping = {"Slow": 0.5, "Normal": 0.2, "Fast": 0.05}

st.title("Real-Time Interactive Sorting Algorithm Visualizer")

st.plotly_chart(display_time_complexity_graph(), use_container_width=True)

sorting_algorithms = {
    "Insertion Sort": insertion_sort_yield,
    "Selection Sort": selection_sort,
    "Bubble Sort": bubble_sort,
    "Merge Sort": merge_sort,
    "Quick Sort": quick_sort,
    "Heap Sort": heap_sort,
    "Counting Sort": counting_sort,
    "Radix Sort": radix_sort,
    "Bucket Sort": bucket_sort  
}

algorithm = st.selectbox("Choose a Sorting Algorithm", list(sorting_algorithms.keys()))

st.write(f"**Time Complexity of {algorithm}:** {time_complexities[algorithm]}")

st.write("Define the array to be sorted:")

input_type = st.radio("Array Input Type", ("Random Array", "Custom Array"))

if input_type == "Random Array":
    size = st.slider("Select Array Size", min_value=5, max_value=100, value=20)
    array = np.random.randint(1, 1000, size).tolist()
    st.write(f"Generated Array (size {size}):", array)
else:
    custom_array = st.text_input("Enter your custom array (comma-separated integers)", "10, 5, 30, 1, 2, 9, 8")
    array = [int(i) for i in custom_array.split(",")]

speed = st.select_slider("Select Sorting Speed", options=["Slow", "Normal", "Fast"], value="Normal")
selected_speed = speed_mapping[speed]

if st.button("Sort Now"):
    st.write("Sorting in progress...")
    start_time = time.time()

    sorting_gen = sorting_algorithms[algorithm]
    
    sorted_array = real_time_sorting_visualization(array, sorting_gen, algorithm, selected_speed)

    if sorted_array:
        end_time = time.time()
        
        st.success("Sorting complete!")
        st.write("Sorted Array:", sorted_array) 
        st.write(f"Time taken: {round((end_time - start_time) * 1000, 2)} ms")
        
    elif sorted_array is None:
        end_time = time.time()
        
        st.success("Sorting complete!")
        st.write("Sorted Array:", array) 
        st.write(f"Time taken: {round((end_time - start_time) * 1000, 2)} ms")

if st.button("Reset"):
    pyautogui.hotkey("ctrl", "F5")
