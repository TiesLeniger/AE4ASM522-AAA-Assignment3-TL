import numpy as np
import matplotlib.pyplot as plt

class TangDowellWing:
    """
    Class that represents a high aspect ratio wing that was used to study aeroelastic responses in
    a study by D. Tang and E.H. Dowell
    """
    def __init__(self, n_c: int, n_s: int, spacing_c: str, spacing_s: str):
        # vlm discretisation parameters
        self.n_c = n_c                                                      # number of chordwise panels
        self.n_s = n_s                                                      # number of spanwise panels
        
        # geometric parameters
        self.semi_span = 0.451                              # [m] semi span of the wing
        self.chord = 0.051                                  # [m] wing chord (constant along span)
        self.beam_mass = 0.2351                             # [kg/m] beam mass per unit length
        self.I_p = 0.2056e-4                                # [kg m^2/m] polar moment of inertia scaled with beam density per unit length (rho * I_p)
        self.d = 0.000508                                   # [m] distance from cross-sectional CG to the elastic axis, positive when CG is more aft than EA

        # create discretisation (needs geometric params)
        self.wingpoints = self._generate_point_mesh(spacing_c, spacing_s)   # generate point mesh of the wing upon initialisation
        self._panel_information()                                           # calculates attributes for panel information

        # structural parameters
        self.EI = 0.4186                                    # [Nm^2] bending stiffness
        self.GJ = 0.9539                                    # [Nm^2] torsional stiffness
        self.K_BT = 0.0                                     # [Nm^2] bend-twist coupling coefficient (KBT**2 < EI*GJ)
        self.zeta_B = 0.02                                  # [-] structural damping in bending, cB/cB_crit
        self.zeta_T = 0.031                                 # [-] structural damping in torsion, cT/cT_crit

        # discrete mass object
        self.dmo_ay = 1                                     # [-] fraction of the span where the dmo is located
        self.dmo_mass = 0.0417                              # [kg] mass of the discrete mass object
        self.dmo_d = 0.0                                    # [m] distance of the discrete mass object to the elastic axis
        self.dmo_I = 0.9753e-4                              # [kg m^2] dmo moment of inertia around the cg

    def _generate_point_mesh(self, spacing_c: str, spacing_s: str) -> np.ndarray:
        """
        Generate a point mesh of the wing
        """
        # discretisation in x
        if spacing_c == "cosine":
            beta = np.linspace(0.0, np.pi, num = self.n_c + 1)              # n_c + 1 because n+1 rows of points make n chordwise elements
            x = self.chord*((1-np.cos(beta))/2)                             # generates higher panel density near LE and TE where higher gradients are present
        elif spacing_c == "constant":
            x = np.linspace(0.0, self.chord, num = self.n_c + 1)            # constant spacing
        else:
            raise ValueError(f"Parameter `spacing_c` can be 'constant' or 'cosine', got: {spacing_c}")
        # discretisation in y
        if spacing_s == "cosine":
            beta = np.linspace(np.pi/2, 0.0, num = self.n_s + 1)
            y = self.semi_span*np.cos(beta)
        elif spacing_s == "constant":
            y = np.linspace(0.0, self.semi_span, num = self.n_s + 1)
        else:
            raise ValueError(f"Parameter `spacing_s` can be 'constant' or 'cosine', got: {spacing_s}")
        # make point mesh 
        points = np.zeros((self.n_c + 1, self.n_s + 1, 3))      # z coordinate remains 0 as the wing airfoil has no camber (NACA 0010)
        points[:, :, 0] = x[:, np.newaxis]                      # broadcast x
        points[:, :, 1] = y[np.newaxis, :]                      # broadcast y
        mirror_points = np.copy(points)
        mirror_points[:, :, 1] *= -1
        mirror_points = np.flip(mirror_points, axis = 1)
        points = np.concatenate((mirror_points[:, :-1, :], points), axis = 1)
        self.n_s *= 2

        return points
    
    def _panel_information(self):
        self.panel_corner_points()
        self.vortex_ring_corner_points()
        self.panel_vectors()
        self.panel_centre_of_pressure()
        self.panel_control_point()
        self.panel_width()

    def panel_corner_points(self):

        self.P1s = self.wingpoints[1:, :-1, :]                       # south east corner points of each panel
        self.P2s = self.wingpoints[:-1, :-1, :]                      # south west corner points of each panel
        self.P3s = self.wingpoints[:-1, 1:, :]                       # north west corner points of each panel
        self.P4s = self.wingpoints[1:, 1:, :]                        # north east corner points of each panel

    def vortex_ring_corner_points(self):

        if not all(hasattr(self, attr) for attr in ["P1s", "P2s", "P3s", "P4s"]):
            self.panel_corner_points()

        P2P1 = self.P1s - self.P2s
        self.A = self.P1s + 0.25 * P2P1
        self.B = self.P2s + 0.25 * P2P1
        P3P4 = self.P4s - self.P3s
        self.C = self.P3s + 0.25 * P3P4
        self.D = self.P4s + 0.25 * P3P4

    def panel_vectors(self):

        if not all(hasattr(self, attr) for attr in ["P1s", "P2s", "P3s", "P4s"]):
            self.panel_corner_points()
        
        self.Ak = self.P4s - self.P2s                               # diagonal vector from south west to north east
        self.Bk = self.P3s - self.P1s                               # diagonal vector from south east to north west
        self.Ek = self.P1s - self.P2s                               # vector that lies along the south border of each panel
        self.Fk = self.P4s - self.P1s                               # vector that lies along the east border of each panel
        cross_product = np.cross(self.Ak, self.Bk, axis = 2)
        self.panel_normal = cross_product/np.linalg.norm(cross_product, axis = 2, keepdims = True)
        self.panel_area = 0.5*(np.linalg.norm(np.cross(self.Fk, self.Bk, axis = 2), axis = 2) + np.linalg.norm(np.cross(self.Bk, self.Ek, axis = 2), axis = 2))
        self.wing_area = 2*np.sum(self.panel_area)

    def panel_centre_of_pressure(self):
        BC = self.C - self.B
        self.panel_cop = self.B + 0.5*BC

    def panel_control_point(self):
        avg_panel_height = 0.5*(self.Ek + (self.P4s - self.P3s))
        self.panel_cntrl = self.panel_cop + 0.5*avg_panel_height

    def panel_width(self):
        self.panel_widths = 0.5*((self.P3s[:, :, 1]-self.P2s[:, :, 1])+(self.P4s[:, :, 1]-self.P1s[:, :, 1]))

    def plot_wing(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_wireframe(self.wingpoints[:, :, 0], self.wingpoints[:, :, 1], self.wingpoints[:, :, 2], color='blue')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.set_title('Wing point mesh')

        # Use y (span) as the reference for all axis limits
        Y = self.wingpoints[:, :, 1]
        y_max = Y.max()
        pad = y_max * 0.1

        # Center x and z at their midpoints
        X = self.wingpoints[:, :, 0]
        Z = self.wingpoints[:, :, 2]
        x_mid = 0.5 * (X.min() + X.max())
        z_mid = 0.5 * (Z.min() + Z.max())

        ax.set_xlim(x_mid - y_max - pad, x_mid + y_max + pad)
        ax.set_ylim(-y_max - pad, y_max + pad)
        ax.set_zlim(z_mid - y_max - pad, z_mid + y_max + pad)

        plt.show()