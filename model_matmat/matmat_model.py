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
    R_copy = 0
    R_scale = 0
    R_add = 0
    R_triad = 0
    nthreads = 0


folder = "mac_output"
filename = "stream_1m_%d"

data_size = 1000000 * 8 * 2
rate_list = ""
R_threads = list()
n_threads = [0, 1, 2, 4]
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
                if rate > rate_list.R_copy:
                    rate_list.R_copy = rate

            elif "Scale" in line:
                if rate > rate_list.R_scale:
                    rate_list.R_scale = rate

            elif "Add" in line:
                if rate > rate_list.R_add:
                    rate_list.R_add = rate

            elif "Triad" in line:
                if rate > rate_list.R_triad:
                    rate_list.R_triad = rate
    f.close()

# Rm = R_threads[0] (serial performance)
# omega = R_threads[0] - R_threads[1]  (overhead, diff between serial and single thread)
# tau * n_threads -> rate improvement when increasing threads to 4 (number of cores on my computer)

# Copy : 
B_m = 1.0 / (R_threads[0].R_copy)
o = 1.0 / (R_threads[0].R_copy - R_threads[1].R_copy)
# To find tau, look at difference in rates from 1 thread and n threads
B_1 = 1.0 / (R_threads[1].R_copy)

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
    times.append((B_1  - tau*(nt-1)) * data_size)
print(times)






if 0:
    Rm = (15019.7*MB)*2.0
    Rc = (28679.0*MB)*2.0
    Rf_min = 1.4*GB*8
    Rf_max = 3.9*GB*8
    floprate = 1.0 / Rf_min


    # Methods to calculate the modeled cost of memory/cache accesses and FLOPS
    def mem_t(n_access, data_size):
        return (1.0/Rm)*n_access*data_size

    def cache_t(n_access, data_size):
        return (1.0/Rc)*n_access*data_size

    def flop_t(n_flops):
        return floprate * n_flops


    # Go through different values of N
    # Model the cost of mat-mat multiplication
    # Compare this to the measured cost
    n_list = [25, 100, 1000]
    if 1:
        data_size = 8
        measured = [5.855892e-06, 4.295490e-04, 5.866849e-01]
    else:
        data_size = 4
        measured = [5.842235e-06, 4.095418e-04, 4.511499e-01]
    mem_n = list()
    cache_n = list()
    if 1:
        mem_n = [25**3, 2*100**3, 2*1000**3]
        cache_n = [25**3, 0, 0]
    else:
        mem_n = [2*25**3, 2*100**3, 2*1000**3]
        cache_n = [0, 0, 0]
    flops = [25**3, 100**3, 1000**3]
    x_data = np.arange(len(n_list))
    modeled = list()
    for i in range(len(n_list)):
        T = mem_t(mem_n[i], data_size) + cache_t(cache_n[i], data_size) + flop_t(flops[i])
        modeled.append(T)


    # Plot comparison of Modeled and Measured Times
    if 1:
        plot.barplot(x_data, [measured, modeled], ["Measured", "Modeled"])
        plot.set_scale('linear', 'log')
        plot.add_labels("N", "Time")
        plot.set_xticks(x_data, n_list)
        plot.display_plot()

    # Plot percentage of model cost that is due to 
    # 1. memory accesses
    # 2. cache accesses
    # 3. flops
    if 1:
        mem_l = [mem_t(n, data_size) for n in mem_n]
        cache_l = [cache_t(n, data_size) for n in cache_n]
        flop_l = [flop_t(n) for n in flops]
        mem_p = [(mem_l[i] / modeled[i])*100 for i in range(len(modeled))]
        cache_p = [(cache_l[i] / modeled[i]) * 100 for i in range(len(modeled))]
        flop_p = [(flop_l[i] / modeled[i]) * 100 for i in range(len(modeled))]
        plot.stacked_barplot(x_data, [mem_p, cache_p, flop_p], ["Mem", "Cache", "Flops"])
        plot.add_labels("N", "Percentage of Time")
        plot.set_xticks(x_data, n_list)
        plot.display_plot()


