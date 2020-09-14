import numpy as np
import pyfancyplot.plot as plot


schedules = ["static", "dynamic", "guided", "static8", "dynamic8", "guided8", "static1"]
folder = "bw_output"
filename = "stream_%s_%d"
for schedule in schedules:
    copy_rate_list = list()
    scale_rate_list = list()
    add_rate_list = list()
    triad_rate_list = list()
    n_threads = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    for nt in n_threads:
        fn = filename%(schedule,nt)
        f = open("%s/%s"%(folder, fn))
        for line in f:
            if "Copy:" in line or "Scale:" in line or "Add:" in line or "Triad:" in line:
                if "Copy" in line:
                    rate_list = copy_rate_list
                elif "Scale" in line:
                    rate_list = scale_rate_list
                elif "Add" in line:
                    rate_list = add_rate_list
                elif "Triad" in line:
                    rate_list = triad_rate_list

                line_list = line.rsplit(" ")
                idx = 1
                while (line_list[idx] == ""):
                    idx += 1
                rate = (float)(line_list[idx])
                idx += 1
                while (line_list[idx] == ""):
                    idx += 1
                avg_time = (float)(line_list[idx])

                rate_list.append(rate)

        f.close()

    plot.add_luke_options()
    plot.line_plot(copy_rate_list, n_threads, label = "Copy")
    plot.line_plot(scale_rate_list, n_threads, label = "Scale")
    plot.line_plot(add_rate_list, n_threads, label = "Add")
    plot.line_plot(triad_rate_list, n_threads, label = "Triad")
    plot.add_anchored_legend()
    plot.add_labels("Number of Threads", "Rate (MB/S)")
    plot.save_plot("bw_stream_%s.pdf"%schedule)
