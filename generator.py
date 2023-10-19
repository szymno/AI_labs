import numpy as np

from matplotlib import pyplot as plt


def create_data_class(modes_size: int,
                      sample_size: int,
                      mode_intervals: tuple[tuple[float, float], tuple[float, float]],
                      std_dev_intervals: tuple[tuple[float, float], tuple[float, float]]):
    """
    Creates data class clusters with given mode and standard deviation intervals.
    :param modes_size: Number of clusters
    :param sample_size: Size of each cluster
    :param mode_intervals: Modes are chosen randomly from between those intervals.
                           Format is ((low_x, low_y), (high_x, high_y))
    :param std_dev_intervals: Standard deviations are chosen randomly from between those intervals.
                              Format is ((low_x, low_y), (high_x, high_y))
    :return: An array of points in a format of pairs of coordinates [[x_0, y_0], [x_1, y_1], ...]
    """

    rng = np.random.default_rng()

    modes = rng.uniform(
        low=(mode_intervals[0]),
        high=(mode_intervals[1]),
        size=(modes_size, 2)
    )

    std_devs = np.abs(
        rng.uniform(low=std_dev_intervals[0],
                    high=std_dev_intervals[1],
                    size=(modes_size, 2))
    )

    data_samples = []
    for mode, std_dev in zip(modes, std_devs):
        data_samples.append(rng.normal(loc=mode, scale=std_dev, size=(sample_size, 2)))

    return np.vstack(data_samples)


def plot_points(data_points_0, data_points_1):
    figure, axis = plt.subplots()
    axis.scatter(data_points_0[:, 0], data_points_0[:, 1], label="Class 0")
    axis.scatter(data_points_1[:, 0], data_points_1[:, 1], label="Class 1",)
    return figure, axis


if __name__ == "__main__":
    interval_modes_0 = (
        (-10., -10.),
        (10., 10.)
    )

    interval_st_devs_0 = (
        (0.5, 0.5),
        (1., 1.)
    )

    interval_modes_1 = (
        (-20., -20.),
        (0., 50.)
    )

    interval_st_devs_1 = (
        (0.5, 0.5),
        (3., 4.)
    )

    data_points_0_ = create_data_class(
        modes_size=3,
        sample_size=500,
        mode_intervals=interval_modes_0,
        std_dev_intervals=interval_st_devs_0
    )
    data_points_1_ = create_data_class(
        modes_size=5,
        sample_size=300,
        mode_intervals=interval_modes_1,
        std_dev_intervals=interval_st_devs_1
    )

    fig, ax = plot_points(data_points_0_, data_points_1_)
    plt.show()
