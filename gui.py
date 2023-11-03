import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

from matplotlib import pyplot as plt

from generator import create_data_class, plot_points


root = tk.Tk()
root.title("Point generator")
#  root.geometry('500x500')
#  root.resizable(False, False)
frame = ttk.Frame(root, padding=15)
frame.grid()


def read_entries(entries_dict):
    return (
        (entries_dict["low"]["x"].get(),  entries_dict["low"]["y"].get()),
        (entries_dict["high"]["x"].get(), entries_dict["high"]["y"].get())
    )


def empty_label(x, y, **kwargs):
    return ttk.Label(frame).grid(column=x, row=y, **kwargs)


def labels(start_x, start_y, label_text):
    label = ttk.Label(frame, text=label_text)
    x_label = ttk.Label(frame, text="x")
    y_label = ttk.Label(frame, text="y")
    from_label = ttk.Label(frame, text="low")
    to_label = ttk.Label(frame, text="high")

    label.grid(column=start_x + 1, row=start_y, columnspan=3)
    x_label.grid(column=start_x, row=start_y + 2)
    y_label.grid(column=start_x, row=start_y + 3)
    from_label.grid(column=start_x + 1, row=start_y + 1)
    to_label.grid(column=start_x + 2, row=start_y + 1)


def block(start_x, start_y, mode, entries_dict):
    labels(start_x, start_y, mode)
    entries_dict["low"]["x"].grid(column=start_x + 1,  row=start_y + 2)
    entries_dict["high"]["x"].grid(column=start_x + 2, row=start_y + 2)
    entries_dict["low"]["y"].grid(column=start_x + 1,  row=start_y + 3)
    entries_dict["high"]["y"].grid(column=start_x + 2, row=start_y + 3)


def plotter(mode_0_dict, mode_1_dict, std_dev_0_dict, std_dev_1_dict, modes_size, sample_size):
    mode_0 = read_entries(mode_0_dict)
    mode_1 = read_entries(mode_1_dict)
    std_dev_0 = read_entries(std_dev_0_dict)
    std_dev_1 = read_entries(std_dev_1_dict)

    try:
        modes_size = int(modes_size)
        sample_size = int(sample_size)
        data_points_0_ = create_data_class(
            modes_size=modes_size,
            sample_size=sample_size,
            mode_intervals=mode_0,
            std_dev_intervals=std_dev_0
        )
        data_points_1_ = create_data_class(
            modes_size=modes_size,
            sample_size=sample_size,
            mode_intervals=mode_1,
            std_dev_intervals=std_dev_1
        )

        fig, ax = plot_points(data_points_0_, data_points_1_)
        fig.legend()
        plt.show()
    except ValueError:
        messagebox.showwarning(title="Wrong data", message="Data was written in a wrong format.\n"
                                                           "Remember that high is greater or equal to low "
                                                           "and standard deviation is a positive number")


mode_0_entries = {
    "low":  {"x": ttk.Entry(frame, width=10), "y": ttk.Entry(frame, width=10)},
    "high": {"x": ttk.Entry(frame, width=10), "y": ttk.Entry(frame, width=10)}
}

mode_1_entries = {
    "low":  {"x": ttk.Entry(frame, width=10), "y": ttk.Entry(frame, width=10)},
    "high": {"x": ttk.Entry(frame, width=10), "y": ttk.Entry(frame, width=10)}
}

st_dev_0_entries = {
    "low":  {"x": ttk.Entry(frame, width=10), "y": ttk.Entry(frame, width=10)},
    "high": {"x": ttk.Entry(frame, width=10), "y": ttk.Entry(frame, width=10)}
}

st_dev_1_entries = {
    "low":  {"x": ttk.Entry(frame, width=10), "y": ttk.Entry(frame, width=10)},
    "high": {"x": ttk.Entry(frame, width=10), "y": ttk.Entry(frame, width=10)}
}


exit_button = ttk.Button(frame, text="Quit", command=root.destroy)

modes_size_label = ttk.Label(frame, text="Number of modes")
modes_size_entry = ttk.Entry(frame, width=10)

samples_size_label = ttk.Label(frame, text="Samples per mode")
samples_size_entry = ttk.Entry(frame, width=10)

plot_button = ttk.Button(
    frame,
    text="Plot graph",
    command=lambda: plotter(
        mode_0_entries,
        mode_1_entries,
        st_dev_0_entries,
        st_dev_1_entries,
        modes_size_entry.get(),
        samples_size_entry.get()
    )
)


block(0, 0, "Class 0", mode_0_entries)
empty_label(0, 4, ipadx=10, ipady=10)
block(0, 5, "Class 1", mode_1_entries)

empty_label(4, 0, padx=10)
block(5, 0, "Std dev of class 0", st_dev_0_entries)
block(5, 5, "Std dev of class 1", st_dev_1_entries)

empty_label(0, 9, pady=10)

modes_size_label.grid(column=0, row=10, columnspan=3)
samples_size_label.grid(column=6, row=10, columnspan=3)

modes_size_entry.grid(column=0, row=11, columnspan=3)
samples_size_entry.grid(column=6, row=11, columnspan=3)

plot_button.grid(column=0, row=13, columnspan=10)


root.mainloop()
