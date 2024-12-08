import numpy as np
import meshio
import matplotlib.pyplot as plt

class Beam:
	def __init__(self, L = 10.0, W = 1.0, H = 1.0, E=10**7, G =3.8*10**6,
				rho = 1.0, nu = 0.3):
		self.L = L  # meters
		self.W = W  # meters
		self.H = H  # meters
		self.E = E  # Elastic Modulus
		self.G = G  # Shear Modulus
		self.rho = rho # mass density
		self.nu = nu
		self.mesh = self.make_mesh()
		
		
	def make_mesh(self, nx=1, ny=1):
		# Generate the grid points
		x = np.linspace(0, self.L, nx+1)
		y = np.linspace(0, self.H, ny+1)
		X, Y = np.meshgrid(x, y)
		points = np.vstack([X.ravel(), Y.ravel()]).T
		
		# Generate connectivity (quad elements)
		cells = []
		for j in range(ny):
			for i in range(nx):
				n1 = j * (nx + 1) + i
				n2 = n1 + 1
				n3 = n1 + (nx + 1)
				n4 = n3 + 1
				cells.append([n1, n2, n4, n3])
	
		cells = np.array(cells)
		
		# Assign boundary codes
		boundary_codes = {0: [], 1: [], 2: [], 3: []} 
		# 0 = Left, 1 = Bottom, 2 = Right, 3 = Top
		
		for idx, (x,y) in enumerate(points):
			if np.isclose(x,0): # Left Bound
				boundary_codes[0].append(idx)
			if np.isclose(y,0): # Bottom Bound
				boundary_codes[1].append(idx)
			if np.isclose(x, self.L): # Right Bound
				boundary_codes[2].append(idx)
			if np.isclose(y, self.H): # Top Bound
				boundary_codes[3].append(idx)
		
		mesh = {
		"points" : points,
		"cells" : cells,
		"nx" : nx,
		"ny" : ny,
		"boundary_codes" : boundary_codes,
		}
		return mesh
	
	def get_dofs_from_boundary_code(self, boundary_code):
		"""Return the DOFs corresponding to a boundary code."""
		boundary_nodes = self.mesh["boundary_codes"][boundary_code]
		dofs = []
		for node in boundary_nodes:
			dofs.append(node*2)
			dofs.append(node*2 + 1)
		return dofs
	
	def refine_mesh(self, nx=1, ny=1):
		self.mesh = self.make_mesh(nx, ny)
		
	def write_mesh(self, file_name):
		meshio.write_points_cells(
		"beam_mesh.vtk",
		self.mesh["points"],
		[("quad", self.mesh["cells"])],
		)


	def get_element_stiffness(self, element_nodes):
		"""Compute element stiffness matrix for plane stress."""
		E = self.E
		nu = self.nu
		D = (E / (1 - nu**2)) * np.array([
			[1, nu, 0],
			[nu, 1, 0],
			[0, 0, (1 - nu) / 2]
		])
		
		ke = np.zeros((8, 8))  # 8 DOFs for a quad element
		gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]
		weights = [1, 1]
		
		for xi in gauss_points:
			for eta in gauss_points:
				N, dN_dxi = self.shape_functions(xi, eta)
				J = dN_dxi.T @ element_nodes
				detJ = np.linalg.det(J)
				
				# Debugging: Print the Jacobian and determinant
				print("Jacobian matrix:")
				print(J)
				print("det(J):", detJ)
				
				if np.isclose(detJ, 0):
					print("Warning: Singular Jacobian detected. Skipping element.")
					print("Element nodes:")
					print(element_nodes)
					continue
				
				invJ = np.linalg.inv(J)
				dN_dx = invJ @ dN_dxi.T
				B = self.get_B_matrix(dN_dx)
				ke += B.T @ D @ B * detJ * weights[0] * weights[1]
		
		return ke



	def assemble_global_stiffness(self):
			"""Assemble the global stiffness matrix."""
			n_nodes = self.mesh["points"].shape[0]
			dof_per_node = 2  # 2 DOFs per node (u, v)
			n_dofs = n_nodes * dof_per_node
			K = np.zeros((n_dofs, n_dofs))
			
			for element in self.mesh["cells"]:
				element_nodes = self.mesh["points"][element]
				print(element_nodes)
				ke = self.get_element_stiffness(element_nodes)
				
				# Map local element DOFs to global DOFs
				dof_map = []
				for node in element:
					dof_map.extend([node * 2, node * 2 + 1])
				
				# Assemble into global stiffness matrix
				for i, global_i in enumerate(dof_map):
					for j, global_j in enumerate(dof_map):
						K[global_i, global_j] += ke[i, j]
			
			return K


	def shape_functions(self, xi, eta):
			"""Return shape functions and derivatives w.r.t xi and eta."""
			# Shape functions
			N = np.array([
				(1 - xi) * (1 - eta) / 4,
				(1 + xi) * (1 - eta) / 4,
				(1 + xi) * (1 + eta) / 4,
				(1 - xi) * (1 + eta) / 4
			])
			
			# Derivatives of shape functions w.r.t xi and eta
			dN_dxi = np.array([
				[-(1 - eta) / 4, -(1 - xi) / 4],
				[(1 - eta) / 4, -(1 + xi) / 4],
				[(1 + eta) / 4, (1 + xi) / 4],
				[-(1 + eta) / 4, (1 - xi) / 4]
			])
			
			return N, dN_dxi


	def get_B_matrix(self, dN_dx):
			"""Construct the strain-displacement matrix B."""
			B = np.zeros((3, 8))
			for i in range(4):
				B[0, i * 2] = dN_dx[0, i]
				B[1, i * 2 + 1] = dN_dx[1, i]
				B[2, i * 2] = dN_dx[1, i]
				B[2, i * 2 + 1] = dN_dx[0, i]
			return B
	
	def export_deformed_mesh(self, displacements, scale=1.0, filename="deformed_mesh.vtk"):
		"""
		Update the mesh with displacements and export the deformed structure.
		
		Parameters:
			displacements (ndarray): The displacement vector
			scale (float or double): scaling factor for visualization
			filename (str): path to write file to
		"""
		# getting number of nodes
		num_nodes = self.mesh["points"].shape[0]
		
		# Reshape displacements into (num_nodes, 2) for x, y directions
		displacements = displacements.reshape((num_nodes,2))
		
		# Compute deformed coordinates
		deformed_coordinates = self.mesh["points"] + scale * displacements
		
		# Export the deformed mesh
		meshio.write_points_cells(
			filename,
			deformed_coordinates,
			[("quad", self.mesh["cells"])],
			)

 
# Apply boundary conditions
def apply_natural_boundary_conditions(K, F, constrained_dofs):
    """Modify stiffness matrix and load vector for boundary conditions."""
    for dof in constrained_dofs:
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1  # To maintain numerical stability
        F[dof] = 0
    return K, F


# Solve for displacements
def solve_displacements(K, F):
    """Solve for nodal displacements."""
    return np.linalg.solve(K, F)


# Post-processing
def compute_reactions(K, u, constrained_dofs):
    """Compute reaction forces at constrained DOFs."""
    return K @ u
    
          
if __name__ == "__main__":	
	B = Beam()
	B.refine_mesh(10, 3)
	B.write_mesh("testing.vtk")
	K = B.assemble_global_stiffness()
	F = np.zeros((K.shape[0],1))
	left_dofs = B.get_dofs_from_boundary_code(0)
	right_dofs = B.get_dofs_from_boundary_code(2)
	F[right_dofs[1::2]] = -1000
	Kmod, Fmod = apply_natural_boundary_conditions(K, F, left_dofs)
	u = solve_displacements(Kmod, Fmod)
	B.export_deformed_mesh(u)
	
	
	
