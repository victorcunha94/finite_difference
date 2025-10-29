"""
Solves the transient 2D heat diffusion with Dirichlet Boundary Conditions using petsc4py in PARALLEL.

Consider the 2D heat equation:

    ∂u/∂t = ∂²u/∂x² + ∂²u/∂y²

with Dirichlet Boundary Conditions on a unit square domain

    u(t, x=0, y) = 0, u(t, x=1, y) = 0
    u(t, x, y=0) = 0, u(t, x, y=1) = 0
"""

import petsc4py
import sys
import numpy as np
import matplotlib.pyplot as plt

petsc4py.init(sys.argv)

from petsc4py import PETSc
from mpi4py import MPI

# Parameters
N_POINTS_X = 101
N_POINTS_Y = 101
TIME_STEP_LENGTH = 0.001
N_TIME_STEPS = 100

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Running 2D Heat Equation with {size} processes")
        print(f"Grid: {N_POINTS_X} x {N_POINTS_Y} = {N_POINTS_X * N_POINTS_Y} DOFs")
    
    # Grid parameters
    h = 1.0 / (N_POINTS_X - 1)
    total_dof = N_POINTS_X * N_POINTS_Y
    
    # Create PETSc matrix - NOW IN PARALLEL
    A = PETSc.Mat().create(comm)
    A.setSizes([total_dof, total_dof])
    A.setType('aij')  # Sparse matrix
    A.setUp()
    
    # Matrix coefficients
    alpha = TIME_STEP_LENGTH / h**2
    center_coeff = 1.0 + 4.0 * alpha
    neighbor_coeff = -alpha
    
    # Determine local range for this process
    local_size = total_dof // size
    remainder = total_dof % size
    
    if rank < remainder:
        local_start = rank * (local_size + 1)
        local_end = local_start + local_size + 1
    else:
        local_start = rank * local_size + remainder
        local_end = local_start + local_size
    
    if rank == 0:
        print(f"Assembling matrix in parallel...")
    
    # Assemble local portion of the matrix
    for global_idx in range(local_start, local_end):
        i = global_idx // N_POINTS_Y
        j = global_idx % N_POINTS_Y
        
        # Boundary conditions: Dirichlet
        if (i == 0 or i == N_POINTS_X - 1 or j == 0 or j == N_POINTS_Y - 1):
            A.setValue(global_idx, global_idx, 1.0)
        else:
            # Interior point - 5-point stencil
            A.setValue(global_idx, global_idx, center_coeff)
            
            # Left neighbor (i-1, j)
            if i > 0:
                left_idx = (i-1) * N_POINTS_Y + j
                A.setValue(global_idx, left_idx, neighbor_coeff)
            
            # Right neighbor (i+1, j)
            if i < N_POINTS_X - 1:
                right_idx = (i+1) * N_POINTS_Y + j
                A.setValue(global_idx, right_idx, neighbor_coeff)
            
            # Bottom neighbor (i, j-1)
            if j > 0:
                bottom_idx = i * N_POINTS_Y + (j-1)
                A.setValue(global_idx, bottom_idx, neighbor_coeff)
            
            # Top neighbor (i, j+1)
            if j < N_POINTS_Y - 1:
                top_idx = i * N_POINTS_Y + (j+1)
                A.setValue(global_idx, top_idx, neighbor_coeff)
    
    A.assemblyBegin()
    A.assemblyEnd()
    
    # Create vectors - also parallel
    b = PETSc.Vec().create(comm)
    b.setSizes(total_dof)
    b.setUp()
    
    x = PETSc.Vec().create(comm)
    x.setSizes(total_dof)
    x.setUp()
    
    # Initial condition - hot spot in the center
    # Only rank 0 initializes the entire array, then we scatter
    if rank == 0:
        initial_condition_global = np.zeros((N_POINTS_X, N_POINTS_Y))
        center_i, center_j = N_POINTS_X // 2, N_POINTS_Y // 2
        radius = min(N_POINTS_X, N_POINTS_Y) // 8
        
        for i in range(N_POINTS_X):
            for j in range(N_POINTS_Y):
                dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                if dist <= radius:
                    initial_condition_global[i, j] = 1.0 - (dist / radius)**2
        
        ic_flat = initial_condition_global.flatten()
    else:
        ic_flat = None
    
    # Scatter initial condition to all processes
    local_ic = np.zeros(local_end - local_start)
    comm.Scatterv(ic_flat, local_ic, root=0)
    
    # Set local portion of the vector
    for local_idx, global_idx in enumerate(range(local_start, local_end)):
        i = global_idx // N_POINTS_Y
        j = global_idx % N_POINTS_Y
        
        # Apply boundary conditions
        if (i == 0 or i == N_POINTS_X - 1 or j == 0 or j == N_POINTS_Y - 1):
            b.setValue(global_idx, 0.0)
        else:
            b.setValue(global_idx, local_ic[local_idx])
    
    b.assemblyBegin()
    b.assemblyEnd()
    
    # Setup linear solver
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setFromOptions()
    
    if rank == 0:
        chosen_solver = ksp.getType()
        print(f"Solving with: {chosen_solver}")
    
    # Time stepping loop
    for time_step in range(N_TIME_STEPS):
        # Solve linear system
        ksp.solve(b, x)
        
        # Update RHS for next time step
        x.copy(b)
        
        # Apply boundary conditions to RHS
        for global_idx in range(local_start, local_end):
            i = global_idx // N_POINTS_Y
            j = global_idx % N_POINTS_Y
            
            if (i == 0 or i == N_POINTS_X - 1 or j == 0 or j == N_POINTS_Y - 1):
                b.setValue(global_idx, 0.0)
        
        b.assemblyBegin()
        b.assemblyEnd()
        
        # Gather results for visualization (only on rank 0)
        if time_step % 10 == 0 or time_step == N_TIME_STEPS - 1:
            if rank == 0:
                solution_global = np.zeros(total_dof)
            else:
                solution_global = None
            
            local_solution = x.getArray()
            comm.Gatherv(local_solution, solution_global, root=0)
            
            if rank == 0:
                solution_2d = solution_global.reshape((N_POINTS_X, N_POINTS_Y))
                max_temp = np.max(solution_2d)
                min_temp = np.min(solution_2d)
                print(f"Time step {time_step + 1:3d}: max T = {max_temp:.6f}, min T = {min_temp:.6f}")
                
                # Visualization (only on rank 0)
                if time_step % 20 == 0:
                    plt.figure(figsize=(8, 6))
                    plt.imshow(solution_2d.T, extent=[0, 1, 0, 1], origin='lower', 
                              cmap='hot', vmin=0, vmax=1)
                    plt.title(f'2D Heat Diffusion - Time step {time_step + 1}')
                    plt.colorbar(label='Temperature')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.tight_layout()
                    plt.savefig(f'heat_2d_step_{time_step+1:03d}.png')
                    plt.close()
    
    if rank == 0:
        print("Simulation completed!")
        print("Use: ffmpeg -framerate 5 -i heat_2d_step_%03d.png heat_diffusion.mp4")
        print("to create a video of the simulation")

if __name__ == "__main__":
    main()
