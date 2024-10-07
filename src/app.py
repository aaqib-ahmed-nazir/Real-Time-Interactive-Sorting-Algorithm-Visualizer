import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go
from function import *

speed_mapping = {"Slow": 0.5, "Normal": 0.2, "Fast": 0.05}

time_complexities = {
    "Insertion Sort": "O(n^2)",
    "Selection Sort": "O(n^2)",
    "Bubble Sort": "O(n^2)",
    "Merge Sort": "O(n log n)",
    "Quick Sort": "O(n log n)",
    "Heap Sort": "O(n log n)",
    "Counting Sort": "O(n + k)",
    "Radix Sort": "O(nk)",
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

    real_time_sorting_visualization(array, sorting_gen, algorithm, selected_speed)

    end_time = time.time()

    st.success("Sorting complete!")
    st.write("Sorted Array:", array)
    st.write(f"Time taken: {round((end_time - start_time) * 1000, 2)} ms")

if st.button("Reset"):
    st.experimental_rerun()
