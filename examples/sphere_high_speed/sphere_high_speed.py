import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np
from sympy import Symbol, sqrt, Max

import modulus
import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint
)

from modulus.sym.domain.inferencer import VoxelInferencer
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.key import Key
from modulus.sym.eq.pdes.navier_stokes import GradNormal, Euler
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.geometry.tessellation import Tessellation
from modulus.sym import quantity
from modulus.sym.eq.non_dim import NonDimensionalizer, Scaler

import stl
from stl import mesh


from sympy import Symbol

from modulus.sym.eq.pde import PDE


class Magnityde(PDE):
    def __init__(self):
        u_scaled = Symbol("u_scaled")
        v_scaled = Symbol("v_scaled")
        w_scaled = Symbol("w_scaled")

        self.equations = {}
        self.equations["magnityte_scaled"] = (
            u_scaled ** 2 + v_scaled ** 2 + w_scaled**2)**(0.5)


@modulus.sym.main(config_path="conf", config_name="conf")
def run(cfg: ModulusConfig) -> None:
    # print(to_yaml(cfg))

    # path definitions
    point_path = to_absolute_path("./stl_files")
    path_inlet = point_path + "/Inlet_large.stl"
    dict_path_outlet = {'path_outlet': point_path + "/Outlet_large.stl",
                        }
    path_noslip = point_path + "/sphere.stl"
    path_interior = point_path + "/closed_large.stl"
    path_outlet_combined = point_path + '/Outlet_large.stl'
    refinment_path = point_path + '/refinment.stl'

    # create and save combined outlet stl
    def combined_stl(meshes, save_path="./combined.stl"):
        combined = mesh.Mesh(np.concatenate([m.data for m in meshes]))
        combined.save(save_path, mode=stl.Mode.ASCII)

    meshes = [mesh.Mesh.from_file(file_)
              for file_ in dict_path_outlet.values()]
    combined_stl(meshes, path_outlet_combined)

    # read stl files to make geometry
    inlet_mesh = Tessellation.from_stl(path_inlet, airtight=False)
    dict_outlet = {}
    for idx_, key_ in enumerate(dict_path_outlet):
        dict_outlet['outlet'+str(idx_)+'_mesh'] = Tessellation.from_stl(
            dict_path_outlet[key_], airtight=False)
    noslip_mesh = Tessellation.from_stl(path_noslip, airtight=True)
    interior_mesh = Tessellation.from_stl(path_interior, airtight=True)
    refinment_mesh = Tessellation.from_stl(refinment_path, airtight=True)

    rho = quantity(1.225, "kg/m^3")
    D = quantity(25, "m")
    inlet_u = quantity(396.0, "m/s")
    inlet_v = quantity(0.0, "m/s")
    inlet_w = quantity(0.0, "m/s")
    inlet_p = quantity(10 ** 5, "pa")

    nd = NonDimensionalizer(
        length_scale=D,
        time_scale=D / inlet_u,
        mass_scale=rho * ((D / 2)**3) * 4 / 3 * 3.1415,
    )

    # normalize meshes
    def normalize_mesh(mesh, center, scale):
        mesh.translate([-c for c in center])
        mesh.scale(scale)

    # normalize invars
    def normalize_invar(invar, center, scale, dims=2):
        invar["x"] -= center[0]
        invar["y"] -= center[1]
        invar["z"] -= center[2]
        invar["x"] *= scale
        invar["y"] *= scale
        invar["z"] *= scale
        if "area" in invar.keys():
            invar["area"] *= scale ** dims
        return invar

    # geometry scaling

    print(nd.ndim(D))
    scale = 1  # / nd.ndim(D)  # turn off scaling

    # center of overall geometry
    center = (0, 0, 0)
    print('Overall geometry center: ', center)

    # center of inlet in original coordinate system
    inlet_center_abs = (0, 0, 0)
    print("inlet_center_abs:", inlet_center_abs)

    # scale end center the inlet center
    inlet_center = list(
        (np.array(inlet_center_abs) - np.array(center)) * scale)
    print("inlet_center:", inlet_center)

    # inlet normal vector; should point into the cylinder, not outwards
    inlet_normal = (1, 0, 0)
    print("inlet_normal:", inlet_normal)

    # inlet velocity profile

    def circular_parabola(x, y, z, center, normal, radius, max_vel):
        centered_x = x - center[0]
        centered_y = y - center[1]
        centered_z = z - center[2]

        distance = sqrt(centered_x ** 2 + centered_y ** 2 + centered_z ** 2)
        parabola = max_vel * Max((1 - (distance / radius) ** 2), 0)
        return normal[0] * parabola, normal[1] * parabola, normal[2] * parabola

    # make aneurysm domain
    domain = Domain()

    # make list of nodes to unroll graph on
    es = Euler(rho="rho", dim=3, time=False)
    navier_stokes_nodes = es.make_nodes()  # ns.make_nodes() + ze.make_nodes()
    gn_p = GradNormal("p", dim=3, time=False)
    gn_u = GradNormal("u", dim=3, time=False)
    gn_v = GradNormal("v", dim=3, time=False)
    gn_w = GradNormal("w", dim=3, time=False)
    gn_rho = GradNormal("rho", dim=3, time=False)
    normal_dot_vel = NormalDotVec(["u", "v", "w"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p"), Key("rho")],
        cfg=cfg.arch.fully_connected,
    )

    mg = Magnityde()

    nodes = (
        navier_stokes_nodes
        + gn_p.make_nodes()
        + gn_u.make_nodes()
        + gn_v.make_nodes()
        + gn_w.make_nodes()
        + gn_rho.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
        + Scaler(
            ["u", "v", "w", "p"],
            ["u_scaled", "v_scaled", "w_scaled", "p_scaled"],
            ["m/s", "m/s", "m/s", "m^2/s^2"],
            nd,
        ).make_node()
        + mg.make_nodes()
    )

    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_mesh,
        outvar={"u": nd.ndim(inlet_u), "v": nd.ndim(inlet_v), "w": nd.ndim(
            inlet_w), "p": nd.ndim(inlet_p), "rho": nd.ndim(rho)},
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    for idx_, key_ in enumerate(dict_outlet):
        outlet = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=dict_outlet[key_],
            # outvar={"normal_gradient_p": 0, "normal_gradient_u": 0, "normal_gradient_v": 0, "normal_gradient_w": 0,"rho" : nd.ndim(rho)},
            outvar={"u": nd.ndim(inlet_u), "v": nd.ndim(inlet_v), "w": nd.ndim(
                inlet_w), "p": nd.ndim(inlet_p), "rho": nd.ndim(rho)},
            batch_size=cfg.batch_size.outlet,
        )
        domain.add_constraint(outlet, "outlet"+str(idx_))

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=noslip_mesh,
        outvar={"normal_gradient_p": 0, "normal_gradient_u": 0,
                "normal_gradient_v": 0, "normal_gradient_w": 0},
        batch_size=cfg.batch_size.no_slip,
        fixed_dataset=True
    )
    domain.add_constraint(no_slip, "no_slip")
    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"continuity": 0, "momentum_x": 0,
                "momentum_y": 0, "momentum_z": 0, "energy": 0},
        batch_size=cfg.batch_size.interior,
        fixed_dataset=True
    )
    domain.add_constraint(interior, "interior")

    refinment = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=refinment_mesh,
        outvar={"continuity": 0, "momentum_x": 0,
                "momentum_y": 0, "momentum_z": 0, "energy": 0},
        batch_size=cfg.batch_size.refinment,
        fixed_dataset=True
    )
    domain.add_constraint(refinment, "refinment")

    voxel_grid = VoxelInferencer(
        bounds=[[-7 * scale, 7 * scale],
                [-7 * scale, 7 * scale], [-7 * scale, 7 * scale]],
        npoints=[128, 128, 128],
        nodes=nodes,
        output_names=["u", "v", "w", "p",
                      "u_scaled", "v_scaled", "w_scaled", "p_scaled", "magnityte_scaled", "rho"]
    )

    domain.add_inferencer(voxel_grid, 'voxel_inf')

    force = PointwiseMonitor(
        noslip_mesh.sample_boundary(900),
        output_names=["p_scaled"],
        metrics={
            "force_x": lambda var: torch.sum(var["normal_x"] * var["area"] * var["p_scaled"]),
            "force_y": lambda var: torch.sum(var["normal_y"] * var["area"] * var["p_scaled"]),
            "force_z": lambda var: torch.sum(var["normal_z"] * var["area"] * var["p_scaled"]),
        },
        nodes=nodes,
    )
    domain.add_monitor(force)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
