"""FEMTIC control file and job-script writers

Module writes the *non-data* FEMTIC inputs and slurm cluster job
scripts.

@author: oaazeved

"""

from pathlib import Path

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from .responses import apply_error_floor


def plot_all_AR(mt_df, title):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    alpha = 0.1
    for i, s in enumerate(mt_df['station'].unique()):
        station = mt_df[mt_df['station']==s].sort_values(by=['period'], ascending=False).reset_index()
        ax1.scatter(station['period'], station['res_yx'], label=f'{s}, Zyx', color='blue', alpha=alpha)
        ax1.scatter(station['period'], station['res_xy'], label=f'{s}, Zxy', color='red', alpha=alpha)
        ax2.scatter(station['period'], station['res_xx'], label=f'{s}, Zxx', color='pink', alpha=alpha)
        ax2.scatter(station['period'], station['res_yy'], label=f'{s}, Zyy', color='green', alpha=alpha)
        ax3.scatter(station['period'], 180+station['phase_yx'], label=f'{s}, Zyx', color='blue', alpha=alpha)
        ax3.scatter(station['period'], station['phase_xy'], label=f'{s}, Zxy', color='red', alpha=alpha)
        ax4.scatter(station['period'], station['phase_xx'], label=f'{s}, Zxx', color='pink', alpha=alpha)
        ax4.scatter(station['period'], station['phase_yy'], label=f'{s}, Zyy', color='green', alpha=alpha)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax3.set_xscale('log')
    ax4.set_xscale('log')
    ax1.set_ylabel('App. Resistivity [Ohm-m]')
    ax2.set_ylabel('App. Resistivity [Ohm-m]')
    ax3.set_ylabel('Phase [deg]')
    ax4.set_ylabel('Phase [deg]')
    ax1.set_xlabel('Period [s]')
    ax2.set_xlabel('Period [s]')
    ax3.set_xlabel('Period [s]')
    ax4.set_xlabel('Period [s]')
    fig.suptitle(title)
    fig.set_layout_engine('tight')
    return


def plot_each_AR(mt_df):

    alpha = 1.0
    for i, s in enumerate(mt_df['station'].unique()):
        fig = plt.figure(figsize=(12, 9))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        station = mt_df[mt_df['station']==s].sort_values(by=['period'], ascending=False).reset_index()
        station = apply_error_floor(station, ['AR'], error_floor_AR=0.1)
        ax1.errorbar(station['period'], station['res_yx'], yerr=station['res_yx_error'], marker='o', label=f'YX', color='blue', alpha=alpha, capsize=3, linewidth=0)
        ax1.errorbar(station['period'], station['res_xy'], yerr=station['res_xy_error'], marker='o', label=f'XY', color='red', alpha=alpha, capsize=3, linewidth=0)
        ax2.errorbar(station['period'], station['res_xx'],  yerr=station['res_xx_error'], marker='o', label=f'XX', color='pink', alpha=alpha, capsize=3, linewidth=0)
        ax2.errorbar(station['period'], station['res_yy'], yerr=station['res_yy_error'], marker='o', label=f'YY', color='green', alpha=alpha, capsize=3, linewidth=0)
        ax3.errorbar(station['period'], 180+station['phase_yx'], yerr=station['phase_yx_error'], marker='o', label=f'YX', color='blue', alpha=alpha, capsize=3, linewidth=0)
        ax3.errorbar(station['period'], station['phase_xy'], yerr=station['phase_yx_error'], marker='o', label=f'XY', color='red', alpha=alpha, capsize=3, linewidth=0)
        ax4.errorbar(station['period'], station['phase_xx'], yerr=station['phase_yx_error'], marker='o', label=f'XX', color='pink', alpha=alpha, capsize=3, linewidth=0)
        ax4.errorbar(station['period'], station['phase_yy'], yerr=station['phase_yx_error'], marker='o', label=f'YY', color='green', alpha=alpha, capsize=3, linewidth=0)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax3.set_xscale('log')
        ax4.set_xscale('log')
        ax1.set_ylabel('App. Resistivity [Ohm-m]')
        ax2.set_ylabel('App. Resistivity [Ohm-m]')
        ax3.set_ylabel('Phase [deg]')
        ax4.set_ylabel('Phase [deg]')
        ax1.set_xlabel('Period [s]')
        ax2.set_xlabel('Period [s]')
        ax3.set_xlabel('Period [s]')
        ax4.set_xlabel('Period [s]')
        ax1.set_ylim(0.1, 1e5)
        ax2.set_ylim(0.01, 1e5)
        ax3.set_ylim(-180, 180)
        ax4.set_ylim(-180, 180)
        ax1.set_xlim(0.1, 10000)
        ax2.set_xlim(0.1, 10000)
        ax3.set_xlim(0.1, 10000)
        ax4.set_xlim(0.1, 10000)
        ax1.legend()
        ax2.legend()
        fig.suptitle(s)
        fig.set_layout_engine('tight')
    return


def plot_df(mt_df, title):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    alpha = 0.1
    for i, s in enumerate(mt_df['station'].unique()):
        station = mt_df[mt_df['station']==s].sort_values(by=['period'], ascending=False).reset_index()
        ax1.scatter(station['period'], np.abs(station['z_yx']), label=f'{s}, Zyx', color='blue', alpha=alpha)
        ax1.scatter(station['period'], np.abs(station['z_xy']), label=f'{s}, Zxy', color='red', alpha=alpha)
        ax2.scatter(station['period'], np.abs(station['z_xx']), label=f'{s}, Zxx', color='pink', alpha=alpha)
        ax2.scatter(station['period'], np.abs(station['z_yy']), label=f'{s}, Zyy', color='green', alpha=alpha)
        ax3.scatter(station['period'], np.angle(station['z_yx'])*180/np.pi, label=f'{s}, Zyx', color='blue', alpha=alpha)
        ax3.scatter(station['period'], np.angle(station['z_xy'])*180/np.pi, label=f'{s}, Zxy', color='red', alpha=alpha)
        ax4.scatter(station['period'], np.angle(station['z_xx'])*180/np.pi, label=f'{s}, Zxx', color='pink', alpha=alpha)
        ax4.scatter(station['period'], np.angle(station['z_yy'])*180/np.pi, label=f'{s}, Zyy', color='green', alpha=alpha)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax3.set_xscale('log')
    ax4.set_xscale('log')
    ax1.set_ylabel('App. Resistivity [Ohm-m]')
    ax2.set_ylabel('App. Resistivity [Ohm-m]')
    ax3.set_ylabel('Phase [deg]')
    ax4.set_ylabel('Phase [deg]')
    ax1.set_xlabel('Period [s]')
    ax2.set_xlabel('Period [s]')
    ax3.set_xlabel('Period [s]')
    ax4.set_xlabel('Period [s]')
    fig.suptitle(title)
    fig.set_layout_engine('tight')
    return


def write_sbatch(write_dir, mailuser, account, qos, runprogram, stp=1, jobname='job', time='00-01:00', 
                mempercpu='15GB', nodes=1, ntasks=1, mailtype='ALL', 
                slurmsubmitdir='$SLURM_SUBMIT_DIR', filename='makeDHexaMesh', 
                runpath='cluster/path/to/the/program',
                modemvarargs=['-I', 'NLCG', 'modem.rho', 'modem.data', 'control.inv', 'control.fwd', 'modem.cov']):
    # print(f"writing {filename}.sbatch for {runprogram}...")
    with open(str(write_dir)+'/'+filename+".sbatch", 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --account={account}\n")
        f.write(f"#SBATCH --qos={qos}\n")
        f.write(f"#SBATCH --job-name={jobname}\n")
        f.write(f"#SBATCH --time={time}\n")
        f.write(f"#SBATCH --mem-per-cpu={mempercpu}\n")
        f.write(f"#SBATCH --nodes={nodes}\n")
        f.write(f"#SBATCH --ntasks-per-node={ntasks}\n")
        f.write(f"#SBATCH --mail-type={mailtype}\n")
        f.write(f"#SBATCH --mail-user={mailuser}\n\n")
        f.write(f"cd {slurmsubmitdir}\n")

        if runprogram.lower() == 'modem':
            f.write("module purge\nmodule load intel\nmodule load openmpi\n")
        else:
            f.write("module purge\nmodule load intel\nmodule load intel-oneapi-mkl\nmodule load intel-oneapi-mpi\n")

        f.write('echo "prog started at: `date`"\n')
        f.write("echo \"\"\n")
        f.write("echo \"Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID\"\n")
        f.write("echo \"\"\n\n")

        if runprogram == 'makeTetraMesh':
            f.write(f"srun {runpath} -stp {int(stp)}\n\n")
            logger.info(f"note: makeTetraMesh step is {stp}")
        elif runprogram == 'makeDHexaMesh':
            f.write(f"srun {runpath}\n\n")
        elif runprogram.lower() == 'femtic':
            f.write(f"mpiexec.hydra -n {ntasks} {runpath}\n\n")

        elif runprogram.lower() == 'modem':
            #f.write(f"mpiexec.hydra -n {ntasks} {runpath} {} {} {} {} {} \n\n")
            modemargstring = ' '.join(modemvarargs)
            f.write(f"srun {runpath} {modemargstring}\n\n")
        else:
            pass
        f.write("echo \"prog ended at: `date`\"\n")

    logger.info(f"{filename}.sbatch written")
    return


def write_inv_control(filepath=Path.cwd(),
                    mesh_type:int=2, inv_method:int=0, data_space_method:int=1, 
                    num_threads:int=1, fwd_solver:int=0, 
                    mem_limit:int=10000, div_num_rhs_fwd:int=1, div_num_rhs_inv:int=1, 
                    elec_field:int=0, owner_element:int=0, resistivity_bounds:int=0, 
                    small_value:float=1.0e-4, output_param=[0], ofile_type:int=0, move_obs_loc:bool=True,
                    distortion:int=0, trade_off_param=1.0, iteration=(0, 20), converge:float=1.0, 
                    rough_matrix:int=0, output_rough_matrix:bool=False, bottom_resistivity:float=10.0, alpha_weight=[1, 1, 1], 
                    diff_filter=None,
                    decrease_threshold:float=0.001, step_length=[[0.5, 0.1, 0.8], [2], [0.6, 1.2]],
                    retrial:int=5, 
                    app_phs_option:int=0):
    """Writes a control file for FEMTIC inversions. 

    :param filepath: path where control file (always 'control.dat') will be written, defaults to Path.cwd()
    :type filepath: Path|str, optional
    :param mesh_type: determines the type of mesh. must be 
        * 0 (Brick hexahedral element), 
        * 1 (Tetrahedral element), or 
        * 2 (Deformed non-conforming hexahedral element), 
        defaults to 2
    :type mesh_type: int, optional
    :param inv_method: determines the inversion method; must be 
        * 0 (Model space method) or 
        * 1 (Data space method), 
        defaults to 0
    :type inv_method: int, optional
    :param data_space_method: Determines type of roughening matrix used in the data space inversion method; must be 
        * 1 (Inverse of roughening matrix R is used) or 
        * 2 (Inverse of R^TR matrix is used), 
        defaults to 1
    :type data_space_method: int, optional
    :param num_threads: determines number of threads to use with OpenMP parallelization. 
        * For serial calculation or parallel calculation with MPI only, set to 1; 
        * for parallel calculations with OpenMP only or hybrid MPI/OpenMP, can be set to integers >1, 
        defaults to 1
    :type num_threads: int, optional
    :param fwd_solver: determines how the forward solver stores information; must be 
        * 0 (In-core mode; information required by the direct solver is stored in RAM), 
        * 1 (Out-of-core mode is selected when the amount of the memory required by the direct solver is larger than that specified below 'MEM_LIMIT'; otherwise in-core mode is selected), or 
        * 2 (Out-of-core mode; some of the information are stored to and read from temporary data files at hard disk); 
        defaults to 0
    :type fwd_solver: int, optional
    :param mem_limit: maximum amount of memory [MB] that can be used by the forward solver, defaults to 10000 (=10 GB)
    :type mem_limit: int, optional
    :param div_num_rhs_fwd: Division number of the right-hand- side vectors of the linear equation solved for the calculation of the sensitivity matrix. In calculating sensitivity matrix, a linear equation with multiple right-hand-sides is solved. When division number of the right-hand-side vectors is one, a routine of PARDISO is called one time to solve the linear equation. On the other hand, when the division number is more than one, that routine is called the specified times and the number of right-hand-side vectors at each time is inversely proportional to the division number as shown in the next slide. In general, the smaller the division number becomes, the faster the speed for solving a linear equation becomes. However, the smaller the division number is, the more memory is required. Defaults to 1
    :type div_num_rhs_fwd: int, optional
    :param div_num_rhs_inv: Division number of the right-hand- side vectors of the linear equation solved for the calculation of the updates of model parameters. This option is used only when the data- space method is selected. When the data-space method is used, a linear equation with multiple right-hand-sides is solved for the calculation of the updates of model parameters. When division number of the right-hand-side vectors is one, a routine of PARDISO is called one time to solve the linear equation. On the other hand, when the division number is more than one, that routine is called the specified times and the number of right-hand-side vectors at each time is inversely proportional to the division number as shown in the next slide. In general, the smaller the division number becomes, the faster the speed for solving a linear equation becomes. However, the smaller the division number is, the more memory is required. Defaults to 1
    :type div_num_rhs_inv: int, optional
    :param elec_field: Type of the electric field used to calculate response functions. 
        0 Horizontal electric field 
        1 Tangential electric field 
        -1 Type of the electric field of each station is individually selected; When this option is selected, you need to specify the type of the electric field for each station in 'observe.dat'. 
        Defaults to 0
    :type elec_field: int, optional
    :param owner_element: Type of owner element of observation stations. 
        0 Downward element 
        1 Upward element 
        -1 Type of owner element of each station is individually selected. When this option is selected, you need to specify the type of owner element for each station in 'observe.dat'. 
        Defaults to 0
    #TODO: finish documentation
    :type owner_element: int, optional
    :param resistivity_bounds: Type of the method limiting subsurface resistivity to be estimated. Must be 
        * 0 (When a resistivity value is exceeds the upper limit or become less than the lower limit, the resistivity value is forced to be the upper limit or the lower limit, respectively) or 
        * 1 (The method proposed by Kim and Kim (2010) is used; see 'A unified transformation function for lower and upper bounding constraints on model parameters in electrical and electromagnetic inversion'. Journal of Geophysics and Engineering, 8(1), 21-26. https://doi.org/10.1088/1742-2132/8/1/004). 
        Defaults to 0
    :type resistivity_bounds: int, optional
    :param small_value: _description_, defaults to 1.0e-4
    :type small_value: float, optional
    :param output_param: _description_, defaults to [0]
    :type output_param: list | tuple | np.ndarray, optional
    :param ofile_type: _description_, defaults to 0
    :type ofile_type: int, optional
    :param move_obs_loc: _description_, defaults to True
    :type move_obs_loc: bool, optional
    :param distortion: _description_, defaults to 0
    :type distortion: int, optional
    :param trade_off_param: _description_, defaults to 10.0
    :type trade_off_param: float | list | tuple | np.ndarray, optional
    :param iteration: _description_, defaults to (0, 20)
    :type iteration: list | tuple | np.ndarray, optional
    :param converge: _description_, defaults to 1.0
    :type converge: float, optional
    :param rough_matrix: _description_, defaults to 0
    :type rough_matrix: int, optional
    :param output_rough_matrix: _description_, defaults to False
    :type output_rough_matrix: bool, optional
    :param bottom_resistivity: _description_, defaults to 10.0
    :type bottom_resistivity: float, optional
    :param alpha_weight: _description_, defaults to [1, 1, 1]
    :type alpha_weight: list | tuple | np.ndarray, optional
    :param diff_filter: _description_, defaults to None
    :type diff_filter: list | tuple | np.ndarray, optional
    :param decrease_threshold: _description_, defaults to 0.001
    :type decrease_threshold: float, optional
    :param step_length: _description_, defaults to [[0.5, 0.1, 0.8], [2], [0.6, 1.2]]
    :type step_length: list | tuple | np.ndarray, optional
    :param retrial: _description_, defaults to 5
    :type retrial: int, optional
    :param app_phs_option: _description_, defaults to 1
    :type app_phs_option: int, optional
    """

    with open(filepath.joinpath("control.dat"), "w") as f:

        if mesh_type not in (0, 1, 2):
            raise ValueError("'mesh_type' must an integer and must be: \n0 (Brick hexahedral element), \n1 (Tetrahedral element), or \n2 (Deformed non-conforming hexahedral element)")
        f.write(f"MESH_TYPE\n{mesh_type}\n")

        if inv_method not in (0, 1):
            raise ValueError("'inv_method' must an integer and must be: \n0 (Model space method) or \n1 (Data space method)")
        f.write(f"INV_METHOD\n{inv_method}\n")
        if data_space_method not in (1, 2):
            raise ValueError("'data_space_method' must an integer and must be: \n1 (Inverse of roughening matrix R is used) or \n2 (Inverse of R^TR matrix is used)")
        f.write(f"DATA_SPACE_METHOD\n{data_space_method}\n")
        if num_threads < 1:
            raise ValueError("'num_threads' must an integer and must be greater than 0")
        f.write(f"NUM_THREADS\n{num_threads}\n")
        if fwd_solver not in (0, 1, 2):
            raise ValueError("'fwd_solver' must an integer and must be: \n0 (In-core mode; information required by the direct solver is stored in RAM), \n1 (Out-of-core mode is selected when the amount of the memory required by the direct solver is larger than that specified below 'MEM_LIMIT'; otherwise in-core mode is selected), or \n2 (Out-of-core mode; some of the information are stored to and read from temporary data files at hard disk).")
        f.write(f"FWD_SOLVER\n{fwd_solver}\n")

        if mem_limit < 1:
            raise ValueError("'mem_limit' must be a number of megabytes(MB) greater than 0")
        f.write(f"MEM_LIMIT\n{mem_limit}\n")
        if div_num_rhs_fwd < 1:
            raise ValueError("'div_num_rhs_fwd' must an integer and must be greater than 0")
        f.write(f"DIV_NUM_RHS_FWD\n{div_num_rhs_fwd}\n")
        if div_num_rhs_inv < 1:
            raise ValueError("'div_num_rhs_inv' must an integer and must be greater than 0")
        f.write(f"DIV_NUM_RHS_INV\n{div_num_rhs_inv}\n")

        if elec_field not in (-1, 0, 1):
            raise ValueError("'elec_field' must an integer and must be: \n0 (horizontal electric field), \n1 (Tangential electric field), or \n-1 (Type of the electric field of each station is individually selected; must be specified in 'observe.dat')")
        if elec_field == -1:
            raise Warning("The type of the electric field of each station is individually selected; must be specified in 'observe.dat'")
        f.write(f"ELEC_FIELD\n{elec_field}\n")
        if owner_element not in (-1, 0, 1):
            raise ValueError("'owner_element' must an integer and must be: \n0 (downward element), \n1 (upward element), or \n-1 (Type of owner element of each station is individually selected; must be specified in 'observe.dat')")
        if owner_element == -1:
            raise Warning("The type of owner element of each station is individually selected; must be specified in 'observe.dat'")
        f.write(f"OWNER_ELEMENT\n{owner_element}\n")
        if resistivity_bounds not in (0, 1):
            raise ValueError("'resistivity_bounds' must an integer and must be: \n0 (When a resistivity value is exceeds the upper limit or become less than the lower limit, the resistivity value is forced to be the upper limit or the lower limit, respectively) or \n1 (The method proposed by Kim and Kim (2010) is used; see A unified transformation function for lower and upper bounding constraints on model parameters in electrical and electromagnetic inversion. Journal of Geophysics and Engineering, 8(1), 21-26. https://doi.org/10.1088/1742-2132/8/1/004)")
        f.write(f"RESISTIVITY_BOUNDS\n{resistivity_bounds}\n")

        f.write(f"SMALL_VALUE\n{small_value:.1e}\n")
        f.write(f"OUTPUT_PARAM\n{len(output_param)}\n")
        if len(output_param)>=1 and len(output_param)<=6:
            for output in output_param:
                if output<0 or output>5:
                    raise ValueError("'output_param' must be between 0 and 5, which are: \n0 (Resistivity), \n1 (Electric field), \n2 (Magnetic Field), \n3 (Electric current), \n4 (Sensitivity), \n5 (Sensitivity density)")
                if output==1:
                    raise Warning("'output_param' includes 1, which is electric field. This will output 4 files per frequency per iteration, which may consume a lot of disk space and take a long time.")
                elif output==2:
                    raise Warning("'output_param' includes 2, which is magnetic field. This will output 4 files per frequency per iteration, which may consume a lot of disk space and take a long time.")
                elif output==3:
                    raise Warning("''output_param' includes 3, which is electric current. This will output 4 files per frequency per iteration, which may consume a lot of disk space and take a long time.")
                else:
                    pass
                f.write(f"{output}\n")
        else:
            if len(output_param)>6:
                raise ValueError("'output_param' must be less than 7 elements long")
        
        if ofile_type not in (0, 1):
            raise ValueError("'ofile_type' must an integer and must be: \n0 (VTK file format [ASCII]) or \n1 (Ensight Gold file format [binary])")
        f.write(f"OFILE_TYPE\n{ofile_type}\n")
        if move_obs_loc:
            f.write(f"MOVE_OBS_LOC\n")

        if distortion not in (0, 1, 2, 3):
            raise ValueError("'distortion' must an integer and must be: \n0 (No estimation of distortion)," + 
                            "\n1 (Differences of distortion matrix and unit matrix are estimated)," + 
                            "\n2 (Both gains and rotations are estimated), or" + 
                            "\n3 (Only gains are estimated)")
        f.write(f"DISTORTION\n{distortion}\n")
        if distortion==0:
            f.write(f"TRADE_OFF_PARAM\n{trade_off_param:.1f}\n")
        elif distortion in (1, 3):
            f.write(f"TRADE_OFF_PARAM\n{trade_off_param[0]:.1f} {trade_off_param[1]:.1f}\n")
        elif distortion==2:
            f.write(f"TRADE_OFF_PARAM\n{trade_off_param[0]:.1f} {trade_off_param[1]:.1f} {trade_off_param[2]:.1f}\n")

        if type(iteration) not in (list, tuple, np.ndarray):
            raise TypeError("'iteration' must be a list, tuple, or numpy array")
        if len(iteration)!=2:
            raise ValueError("'iteration' must have 2 elements; the first element is the number of the starting iteration and the second element is the number of the final iteration")
        if iteration[0]<0 or iteration[1]<0:
            raise ValueError("'iteration' elements must be positive integers")
        f.write(f"ITERATION\n{iteration[0]} {iteration[1]}\n")
        if converge<0.0 or converge>=100.0:
            raise ValueError("'converge' must be a real number greater than 0; this is a percent value. If the change rates of the objective function and its respective terms of the current iteration from those of the previous iteration are less than the threshold value, the Gauss- Newton iteration is finished.")
        f.write(f"CONVERGE\n{converge}\n")

        if rough_matrix not in (0, 1, 2, -1):
            raise ValueError("'rough_matrix' must be an integer and must be: \n0 (Based on element), \n1 (based on parameter cell), \n2 (Based on element with weights based on area-volume ratio), or \n -1 (From external file 'roughening_matrix.dat')")
        f.write(f"ROUGH_MATRIX\n{rough_matrix}\n")
        if output_rough_matrix:
            f.write(f"OUTPUT_ROUGH_MATRIX\n")
        if bottom_resistivity<0.0:
            raise ValueError("'bottom_resistivity' must be a real number greater than 0")
        f.write(f"BOTTOM_RESISTIVITY\n{bottom_resistivity:.1f}\n")

        if len(alpha_weight)!=3:
            raise ValueError("'alpha_weight' must have 3 elements")
        if any(alpha_weight)<0.0:
            raise ValueError("'alpha_weight' elements must be real numbers greater than 0")
        f.write(f"ALPHA_WEIGHT\n{alpha_weight[0]} {alpha_weight[1]} {alpha_weight[2]}\n")

        if diff_filter is not None:
            #TODO: finish errors, make sure indexing works
            if diff_filter[0] not in (1, 2):
                raise ValueError("'diff_filter[0]' must be an integer and must be: \n1 (L1 norm) or \n2 (L2 norm)")
            f.write(f"DIFF_FILTER\n{diff_filter[0]}\n{diff_filter[1][0]} {diff_filter[1][1]}\n{diff_filter[2]}\n{diff_filter[3]}\n")

        if decrease_threshold<0.0:
            raise ValueError("'decrease_threshold' must be a real number greater than 0")
        f.write(f"DECREASE_THRESHOLD\n{decrease_threshold}\n")

        #TODO: finish errors, make sure indexing works
        f.write(f"STEP_LENGTH\n{step_length[0][0]:.1f} {step_length[0][1]:.1f} {step_length[0][2]:.1f}\n{int(step_length[1][0])}\n{step_length[2][0]:.1f} {step_length[2][1]:.1f}\n")
        if retrial<0 or type(retrial)!=int:
            raise ValueError("'retrial' must be a non-negative integer")
        f.write(f"RETRIAL\n{retrial}\n")

        if app_phs_option not in (0, 1):
            raise ValueError("'app_phs_option' must be an integer and must be: \n0 (No special treatment), or \n1 (Impedance tensor is used instead of apparent resistivity and phase if the sign of the real part of impedance tensor component is different between observed and calculated responses. This treatment will improve the stabilization of the inversion using apparent resistivity and phase.)")
        f.write(f"APP_PHS_OPTION \n{app_phs_option}\n")
        f.write("END\n")
    logger.info(f"{filepath}/control.dat written")
