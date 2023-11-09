import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from generator import *
from neuron import Neuron, ActivationFunctions


with st.sidebar:
    st.title("DATA GENERATION")
    num_samples = st.slider("Samples per mode", 100, 1000, 500, 50)
    num_modes = st.slider("Modes per class", 1, 10, 2)

    st.divider()

    st.header("Class 0")
    st.subheader("Ranges")
    class_0_X = st.slider("X range", -100, 100, (-100, 100), 5)
    class_0_Y = st.slider("Y range", -100, 100, (-100, 100), 5)

    st.subheader("Standard deviations")
    class_0_std_dev_X = st.slider("X range ", 0., 25., (0.5, 5.), 0.25)
    class_0_std_dev_Y = st.slider("Y range ", 0., 25., (0.5, 5.), 0.25)

    st.divider()

    st.header("Class 1")
    st.subheader("Ranges")
    class_1_X = st.slider("X range  ", -100, 100, (-100, 100), 5)
    class_1_Y = st.slider("Y range  ", -100, 100, (-100, 100), 5)

    st.subheader("Standard deviations")
    class_1_std_dev_X = st.slider("X range   ", 0., 25., (0.5, 5.), 0.25)
    class_1_std_dev_Y = st.slider("Y range   ", 0., 25., (0.5, 5.), 0.25)
    do_create_data = st.button("Generate datapoints")

    st.divider()

    st.title("NEURON")
    neuron_bias = st.slider("Bias", 0., 1., 0.5, 0.05)
    activation_function = st.selectbox(
        "Choose activation function",
        ("Heaviside", "Logistic", "Sin", "TanH", "Sign", "ReLu", "Leaky ReLu"))

    if activation_function == "Logistic":
        beta_logistic = st.slider("β", 0., 1., 0.2, 0.1)
    epoch_range = list(range(0, 1_000_001, 500))
    epoch_range[0] = 1
    epochs = st.select_slider("Epochs", epoch_range, 500)

    is_training_rate_variable = st.toggle("Is the learning rate variable")
    if is_training_rate_variable:
        learning_rate = (st.number_input("Maximum learning rate", min_value=0., value=1e-2, step=1e-6, format="%f"),
                         st.number_input("Minimum learning rate", min_value=0., value=5e-4, step=1e-6, format="%f"))
    else:
        learning_rate = st.number_input("Learning rate", min_value=0., value=1e-3, step=1e-6, format="%f")

    batch_size = st.select_slider("Batch size", [1, 2, 5, 10, 25, 50, 100])

    do_train_neuron = st.button("Train neuron")

st.title("GENERATED DATA")
if do_create_data:
    try:
        modes_size = num_modes
        sample_size = num_samples
        data_points_0 = create_data_class(
            modes_size=modes_size,
            sample_size=sample_size,
            mode_intervals=(
                (class_0_X[0], class_0_Y[0]),
                (class_0_X[1], class_0_Y[1])
            ),
            std_dev_intervals=(
                (class_0_std_dev_X[0], class_0_std_dev_X[0]),
                (class_0_std_dev_X[1], class_0_std_dev_X[1])
            ),
        )
        data_points_1 = create_data_class(
            modes_size=modes_size,
            sample_size=sample_size,
            mode_intervals=(
                (class_1_X[0], class_1_Y[0]),
                (class_1_X[1], class_1_Y[1])
            ),
            std_dev_intervals=(
                (class_1_std_dev_X[0], class_1_std_dev_X[0]),
                (class_1_std_dev_X[1], class_1_std_dev_X[1])
            ),
        )
        fig_0, ax_0 = plot_points(data_points_0, data_points_1)
        st.session_state.generated = [
            modes_size,
            sample_size,
            data_points_0,
            data_points_1,
            (fig_0, ax_0)
        ]
        fig_0.legend()
        st.pyplot(fig_0)
    except ValueError:
        pass
else:
    try:
        modes_size, sample_size, data_points_0, data_points_1, (fig_0, ax_0) = st.session_state.generated
        st.pyplot(fig_0)

    except AttributeError:
        pass

st.title("TRAINED")
if do_train_neuron:
    samples = np.append(data_points_0, data_points_1, axis=0)
    labels = np.append(np.zeros((data_points_0.shape[0], 1)), np.ones((data_points_1.shape[0], 1)), axis=0)

    samples_norm = (samples - np.min(samples)) / (np.max(samples) - np.min(samples))

    ActivationFunction = {"Heaviside": ActivationFunctions.Heaviside,
                          "Logistic": ActivationFunctions.Logistic,
                          "Sin": ActivationFunctions.Sin,
                          "TanH": ActivationFunctions.TanH,
                          "Sign": ActivationFunctions.Sign,
                          "ReLu": ActivationFunctions.ReLu,
                          "Leaky ReLu": ActivationFunctions.LeakyReLu}

    neuron = Neuron(2, ActivationFunction[activation_function], neuron_bias)
    neuron.train(samples_norm, labels, batch_size, learning_rate=learning_rate)

    max_x, min_x = np.max(samples[:, 0]), np.min(samples[:, 0])
    max_y, min_y = np.max(samples[:, 1]), np.min(samples[:, 1])
    y1, y2 = neuron.weights[0, 0] * min_x + neuron.weights[0, 1], neuron.weights[0, 0] * max_x + neuron.weights[0, 1]

    fig_1, ax_1 = plot_points(data_points_0, data_points_1)
    st.text(f"weights: {neuron.weights[0]}")

    ax_1.fill_between((min_x, max_x, min_x, min_x), (y1, y2, max_y, min_y), facecolor='red', alpha=0.3)
    ax_1.fill_between((min_x, max_x, max_x, min_x), (y1, min_y, y2, y1), facecolor='green', alpha=0.3)

    ax_1.set_xlim([min_x, max_x])
    ax_1.set_ylim([min_y, max_y])

    ax_1.axline((max_x, y2), (min_x, y1))
    st.session_state.generated_train = [
        neuron,
        (fig_1, ax_1)
    ]
    st.pyplot(fig_1)
else:
    try:
        neuron, (fig_1, ax_1) = st.session_state.generated_train
        st.pyplot(fig_1)
    except AttributeError:
        pass




