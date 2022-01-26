import os

def create_dirs(N_ori, N, inc, obl, prot, veq):

    ori_base = "Spots/Orientation {0}/".format(N_ori)

    if not os.path.isdir(ori_base):
        os.mkdir(ori_base)

    ori_info = ori_base + "N{0:d}_inc{1:.1f}_obl{2:.1f}_P{3:.1f}.txt".format(N, inc, obl, prot)
    if not os.path.isfile(ori_info):
        open(ori_info, 'w').close()

    sim_data_path = ori_base + "Simulated Data/"
    op_kern_param_path = ori_base + "Optimal Kernel Parameters/"
    dia_plot_path = ori_base + "Diagnostic Plots/"
    trace_path = ori_base + "Traces/"
    trace_post_path = ori_base + "Trace Posteriors/"
    kern_para_comp_path = ori_base + "Kernel Parameter Comparison/"

    paths = [sim_data_path, op_kern_param_path, dia_plot_path,
             trace_path, trace_post_path, kern_para_comp_path]

    for p in paths:
        if not os.path.isdir(p):
            os.mkdir(p)

    return paths
