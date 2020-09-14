# To run this script, you need to have installed:
# number, matplotlib, pyfancyplot, pandas, and seaborn
# All of these are available with pip install

import numpy as np
import matplotlib.pyplot as plt
import pyfancyplot.plot as plot


#MB = 1024*1024;
#GB = MB*1024;
MB = 1000000;
GB = MB * 1000;


class Rate():
    copy = ""
    scale = ""
    add = ""
    triad = ""

    def __init__(self):
        self.copy = list()
        self.scale = list()
        self.add = list()
        self.triad = list()


folder = "mac_output"
filename = "stream_1m_%d"

data_size = 1000000 * 8 * 2
rate_list = ""
R_threads = list()
n_threads = [0, 1, 2, 3, 4, 6, 8, 10]
for nt in n_threads:
    fn = filename%nt
    f = open("%s/%s"%(folder, fn))
    R_threads.append(Rate())
    rate_list = R_threads[-1]
    for line in f:
        if "Copy:" in line or "Scale:" in line or "Add:" in line or "Triad:" in line:
            line_list = line.rsplit(" ")
            idx = 1
            while (line_list[idx] == ""):
                idx += 1
            rate = (float)(line_list[idx]) * MB

            if "Copy" in line:
                rate_list.copy.append(rate)

            elif "Scale" in line:
                rate_list.scale.append(rate)

            elif "Add" in line:
                rate_list.add.append(rate)

            elif "Triad" in line:
                rate_list.triad.append(rate)

    f.close()


### Plot STREAM Measured Times ###
# time = (1.0 / (rate * 1000000)) * data_size
if 0:
    plot.add_luke_options()
    copy_t = list()
    copy_n = list()
    for n in range(len(n_threads)):
        for r in R_threads[n].copy:
            copy_t.append((1.0 / r))
            copy_n.append(n_threads[n])
    plot.scatter_plot(copy_n, copy_t, label = "Copy")
    plot.display_plot()


# Accessing Main Memory at serial STREAM rate
B_m = 1.0 / (max(R_threads[0].copy))

# Threads have an overhead (initializng OpenMP), a per-thread overhead (initializing threads), and a per-thread bandwidth overhead (more expensive to read and write with multiple threads)
times = list()
mat = list()
for i in range(1, 4):
    times.append(1.0 / max(R_threads[i].copy))
    mat.append([i, data_size/i])
A = np.matrix(mat)
b = np.array(times)
o, B_o = np.linalg.lstsq(A, b)[0]
print(o, B_o)

times = list()
for nt in n_threads[1:5]:
    times.append(o*nt + (B_m + B_o) * (data_size / nt))
print(times)

if 0:

    # Copy : 
    B_m = 1.0 / (R_threads[0].R_copy)
    B_1 = 1.0 / (R_threads[1].R_copy)

    # Overhead from using timing
    # Not dependent on data_size, just a set overhead
    o = (B_m - B_1) * data_size 

    # To find tau, look at difference in rates from 1 thread and n threads
    times = list()
    nt = list()
    for i in range(2, len(n_threads)):
        times.append(B_1 - 1.0 / R_threads[i].R_copy)
        nt.append([n_threads[i]-1])
    A = np.matrix(nt)
    b = np.array(times)
    print(A, b)
    tau, = np.linalg.lstsq(A, b)[0]
    print(tau)

    times = list()
    for nt in n_threads[1:]:
        print((1.0 / (B_1 - tau * (nt-1))) / 1000000)
        times.append((B_m  - tau*(nt-1)) * data_size + o)
    print(times)



