import streamlit as st

from generator import *
from neuron import ActivationFunctions
from neural_net import NeuralNet, Normalizer


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
    st.title("NEURAL NET")
    st.divider()
    st.subheader("Number of layers")
    layers = st.slider("Layers", 1, 10, 1)
    st.divider()
    activation_functions = []
    neurons = []
    betas = []
    for num in range(layers):
        neurons.append(st.slider(f"Neurons in layer {num}", 1, 100, 1))
        activation_functions.append(
            st.selectbox(
                f"Choose activation function for layer {num}",
                ("Logistic", "Heaviside", "Sin", "TanH", "Sign", "ReLu", "Leaky ReLu")
            )
        )

        if activation_functions[num] == "Logistic":
            betas.append(st.slider(f"β for layer {num}", -1., -0.1, -1., 0.1))
        st.divider()

    neurons.append(st.slider("Neurons in output layer", 1, 10, 2))
    activation_functions.append(
        st.selectbox(
            "Choose activation function for output layer",
            ("Logistic", "Heaviside", "Sin", "TanH", "Sign", "ReLu", "Leaky ReLu")
        )
    )
    if activation_functions[num] == "Logistic":
        betas.append(st.slider(f"β for output layer", -1., -0.1, -1., 0.1))

    st.divider()
    st.subheader("NEURAL NET PROPERTIES")
    epoch_range = list(range(0, 1_001, 10))
    epochs = st.select_slider("Epochs", epoch_range, 10)

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
    labels = np.append([1, 0] * np.ones((data_points_0.shape[0], 1)),
                       [0, 1] * np.ones((data_points_1.shape[0], 1)), axis=0)
    normalizer = Normalizer(samples)
    samples_norm = normalizer.minmax(samples)

    ActivationFunction = {"Heaviside": ActivationFunctions.Heaviside,
                          "Logistic": ActivationFunctions.Logistic,
                          "Sin": ActivationFunctions.Sin,
                          "TanH": ActivationFunctions.TanH,
                          "Sign": ActivationFunctions.Sign,
                          "ReLu": ActivationFunctions.ReLu,
                          "Leaky ReLu": ActivationFunctions.LeakyReLu}
    print(neurons + [2])
    neural_net = NeuralNet(neurons, 2, [ActivationFunction[function]() for function in activation_functions])

    with st.spinner("Training"):
        neural_net.train(samples_norm, labels, epochs, learning_rate=learning_rate, batch_size=batch_size)

    fig_1, ax_1 = plot_points(data_points_0, data_points_1)

    X, Y = np.meshgrid(np.linspace(normalizer.base_values["min"][0] - 5, normalizer.base_values["max"][0] + 5, 200),
                       np.linspace(normalizer.base_values["min"][1] - 5, normalizer.base_values["max"][1] + 5, 200))

    X_norm = normalizer.minmax_dim(X, 0)
    Y_norm = normalizer.minmax_dim(Y, 1)

    mesh_points = np.append(X_norm.reshape((-1, 1)), Y_norm.reshape((-1, 1)), axis=1)
    prediction = neural_net.predict(mesh_points)

    ax_1.contourf(X, Y, prediction.reshape(X.shape), alpha=0.25, cmap=plt.cm.RdYlBu)

    st.session_state.generated_train = [
        neural_net,
        (fig_1, ax_1)
    ]
    st.pyplot(fig_1)
else:
    try:

        neural_net, (fig_1, ax_1) = st.session_state.generated_train
        st.pyplot(fig_1)
    except AttributeError:
        pass




