import cvxpy as cp

# Number of casualty zones and hospitals
num_zones = 3
num_hospitals = 3

# Number of people needing transport from each casualty zone
P = [12, 15, 20]

# Capacity of each hospital
C = [20, 30 , 25]

# Total number of available vehicles
V = 15

# Maximum number of people that a vehicle can carry
M = 4

# Cost matrix representing cost of transportation from each casualty zone to each hospital
C_ij = [[3, 2 , 5],
        [3, 4 , 4],
        [2, 9 , 2]]

# Define decision variables
X = cp.Variable((num_zones, num_hospitals), integer=True) # Matrix that defines the number of people Transported from i site to j hospital
Y = cp.Variable((num_zones, num_hospitals), integer=True) # Numbe of trips from i site to J hospital.

# Define objective function
objective = cp.Maximize(cp.sum(X))

# Define constraints
constraints = [
    cp.sum(X, axis=1) <= P,                # Constraint 1: No. of people transported from one zone should be lesser than or equal to max people in that zone 
    cp.sum(X, axis=0) <= C,                # Constraint 2: Capacity constrain in each hospital 
    cp.sum(X) <= V * M,                    # Constraint 3: Transport constraint 
    X <= M * Y,                            # Constraint 4: Number of people that can be transported
    cp.sum(cp.multiply(X, C_ij)) <= 100,   # Constraint 5 (example budget limit)
    X >= 0                                  # x should be non negative                     
]

# Formulate the problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve(solver=cp.GLPK_MI)

# Print the optimal solution
print("Optimal value (number of people saved):", problem.value)
print("Optimal X:")
print(X.value)
print("Optimal Y:")
print(Y.value)