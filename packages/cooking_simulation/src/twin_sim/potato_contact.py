"""Procedural rigid potato and force patches shared by diagnostics and RL.

These are idealized contact-derived tactile signals, not a calibrated skin sensor.
All forces are forces ON the selected patch, expressed in world coordinates.
"""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from twin_sim.paths import scene_path


@dataclass(frozen=True)
class PotatoConfig:
    radii_m: tuple[float, float, float] = (0.060, 0.035, 0.028)
    mass_kg: float = 0.18
    friction: float = 0.65
    shape_seed: int = 0
    bottom_cut_m: float = 0.0

    def __post_init__(self):
        radii = np.asarray(self.radii_m)
        if radii.shape != (3,) or not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError("radii must contain three positive finite values")
        if not np.isfinite(self.bottom_cut_m) or not 0 <= self.bottom_cut_m < radii[2]:
            raise ValueError("bottom_cut_m must be finite, non-negative and below vertical radius")
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0:
            raise ValueError("mass_kg must be positive and finite")
        if not np.isfinite(self.friction) or self.friction < 0:
            raise ValueError("friction must be non-negative and finite")


def _numbers(values) -> str:
    return " ".join(f"{float(value):.10g}" for value in np.asarray(values).flat)


def potato_vertices(config: PotatoConfig) -> np.ndarray:
    """Sample the potato skin, optionally clip a thin lower cap and close flat.

    The cut boundary is interpolated on the sampled surface triangles. MuJoCo
    constructs the convex hull, including the planar cap, for visual and contact.
    """
    rng = np.random.default_rng(config.shape_seed)
    phase = rng.uniform(-np.pi, np.pi, 3)
    points = []
    for theta in np.linspace(0, np.pi, 17):
        for phi in np.linspace(0, 2 * np.pi, 32, endpoint=False):
            direction = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
            # Smooth low-frequency asymmetry; the compiled convex hull is used
            # for BOTH rendering and collision. No hidden sphere proxy.
            radius = 1 + .045*np.sin(3*phi+phase[0])*np.sin(theta)**2 + .035*np.cos(3*theta+phase[1])
            points.append(direction * np.asarray(config.radii_m) * radius)
    points = np.asarray(points)
    if config.bottom_cut_m == 0:
        return points
    plane = float(points[:,2].min()+config.bottom_cut_m)
    boundary = []
    # Include every edge of the latitude/longitude surface triangulation.
    for row in range(16):
        for col in range(32):
            a, b = row*32+col, row*32+(col+1)%32
            c, d = a+32, b+32
            for first, second in ((a,b),(a,c),(a,d),(b,d),(c,d)):
                p, q = points[first], points[second]
                if (p[2]-plane)*(q[2]-plane) < 0:
                    intersection = p+(q-p)*(plane-p[2])/(q[2]-p[2])
                    intersection[2] = plane
                    boundary.append(intersection)
    return np.unique(np.round(np.vstack((points[points[:,2]>=plane],boundary)),12),axis=0)


def add_potato(root: ET.Element, config: PotatoConfig, position) -> None:
    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("position must be a finite 3-vector")
    points = potato_vertices(config)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="potato_mesh", vertex=_numbers(points))
    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", name="potato", pos=_numbers(position))
    ET.SubElement(body, "freejoint", name="potato_free")
    ET.SubElement(body, "geom", name="potato_collision", type="mesh", mesh="potato_mesh",
                  mass=str(config.mass_kg), rgba="0.65 0.43 0.18 1", contype="1", conaffinity="3",
                  friction=f"{config.friction} 0.005 0.0001", condim="6", solref="0.01 1")


def build_contact_probe_xml(config: PotatoConfig = PotatoConfig()) -> str:
    root = ET.fromstring('''<mujoco model="Potato physical contact probe">
      <option timestep="0.002" integrator="implicitfast"/>
      <worldbody><light pos="0 -1 2"/>
        <geom name="probe_table" type="box" pos="0 0 -.01" size="0.5 0.5 .01" rgba=".5 .35 .2 1" friction=".65 .005 .0001"/>
      </worldbody></mujoco>''')
    add_potato(root, config, (0, 0, 0.12))
    return ET.tostring(root, encoding="unicode")


def build_robot_potato_xml(config: PotatoConfig = PotatoConfig(), *,
                           position=(.435, -.012, .26), source: Path | None = None) -> str:
    source = scene_path() if source is None else Path(source).resolve()
    root = ET.parse(source).getroot()
    world = root.find("worldbody")
    # The source also hosts a separate pick-and-place experiment. Omit its
    # pedestals/cube/markers from this task rather than leave obstacles nearby.
    for element in list(world):
        if element.get("name", "").startswith("pick_"):
            world.remove(element)
    # Existing model assets remain external; generated scene works from any cwd.
    for element in root.iter():
        if "file" in element.attrib:
            element.set("file", str((source.parent / element.get("file")).resolve()))
    add_potato(root, config, position)
    return ET.tostring(root, encoding="unicode")


@dataclass(frozen=True)
class ContactPatch:
    normal_force_n: float
    force_world_n: np.ndarray
    center_world_m: np.ndarray
    contact_count: int


def read_contact_patch(model, data, geom_ids) -> ContactPatch:
    selected = set(int(i) for i in geom_ids)
    if any(i < 0 or i >= model.ngeom for i in selected):
        raise ValueError("invalid tactile geom id")
    normal = 0.0
    force = np.zeros(3)
    weighted_center = np.zeros(3)
    count = 0
    wrench = np.zeros(6)
    for index, contact in enumerate(data.contact):
        first, second = int(contact.geom1) in selected, int(contact.geom2) in selected
        if first == second or contact.efc_address < 0:
            continue
        mujoco.mj_contactForce(model, data, index, wrench)
        load = max(float(wrench[0]), 0.0)
        if load <= 0:
            continue
        # MuJoCo reports force on geom2; contact frame axes occupy rows.
        force += (1 if second else -1) * contact.frame.reshape(3, 3).T @ wrench[:3]
        normal += load
        weighted_center += load * contact.pos
        count += 1
    return ContactPatch(normal, force, weighted_center / normal if normal else np.zeros(3), count)


def hand_tactile(model, data) -> np.ndarray:
    """Ten patches: five pads, then five middle phalanges; 8 channels each.

    Channels: normal load [N], local force xyz [N], local contact center xyz [m],
    contact flag. Contact identities are deliberately excluded from observations.
    """
    rows = []
    for suffix in ("pad", "link3_collision"):
        for finger in range(1, 6):
            geom = model.geom(f"left_finger{finger}_{suffix}").id
            body = int(model.geom_bodyid[geom])
            sample = read_contact_patch(model, data, [geom])
            rotation = data.xmat[body].reshape(3, 3)
            center = rotation.T @ (sample.center_world_m - data.xpos[body]) if sample.contact_count else np.zeros(3)
            rows.append(np.r_[sample.normal_force_n, rotation.T @ sample.force_world_n, center, sample.contact_count > 0])
    return np.asarray(rows, dtype=np.float32)
