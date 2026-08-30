from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from twin_sim.names import LEFT_ARM, RIGHT_ARM, ArmNames
from twin_sim.hand_names import HAND_ACTUATORS, HAND_JOINTS
from twin_sim.paths import scene_path


class ModelValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArmIndices:
    joint_ids: np.ndarray
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    actuator_ids: np.ndarray


@dataclass(frozen=True)
class HandIndices:
    joint_ids: np.ndarray
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    actuator_ids: np.ndarray


@dataclass
class SimulationModel:
    model: mujoco.MjModel
    data: mujoco.MjData
    left: ArmIndices
    right: ArmIndices
    hand: HandIndices

    @classmethod
    def load(cls, path: Path | None = None) -> "SimulationModel":
        model_path = scene_path() if path is None else Path(path)
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        simulation = cls(
            model=model,
            data=data,
            left=cls._arm_indices(model, LEFT_ARM),
            right=cls._arm_indices(model, RIGHT_ARM),
            hand=cls._hand_indices(model),
        )
        simulation.require_site("right_tool_tip_site")
        simulation.require_sensor("right_tool_force")
        simulation.require_sensor("right_tool_torque")
        simulation.require_geom("right_knife_blade")
        simulation.require_geom("chopping_board")
        simulation.validate_actuator_contract()
        mujoco.mj_forward(model, data)
        return simulation

    @classmethod
    def _arm_indices(cls, model: mujoco.MjModel, names: ArmNames) -> ArmIndices:
        joint_ids = np.asarray(
            [cls._require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint", name) for name in names.joints],
            dtype=np.int32,
        )
        actuator_ids = np.asarray(
            [
                cls._require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator", name)
                for name in names.actuators
            ],
            dtype=np.int32,
        )
        return ArmIndices(
            joint_ids=joint_ids,
            qpos_ids=model.jnt_qposadr[joint_ids].copy(),
            dof_ids=model.jnt_dofadr[joint_ids].copy(),
            actuator_ids=actuator_ids,
        )

    @classmethod
    def _hand_indices(cls, model: mujoco.MjModel) -> HandIndices:
        joint_ids = np.asarray(
            [
                cls._require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint", name)
                for name in HAND_JOINTS
            ],
            dtype=np.int32,
        )
        actuator_ids = np.asarray(
            [
                cls._require_id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator", name
                )
                for name in HAND_ACTUATORS
            ],
            dtype=np.int32,
        )
        return HandIndices(
            joint_ids=joint_ids,
            qpos_ids=model.jnt_qposadr[joint_ids].copy(),
            dof_ids=model.jnt_dofadr[joint_ids].copy(),
            actuator_ids=actuator_ids,
        )

    @staticmethod
    def _require_id(
        model: mujoco.MjModel, object_type: mujoco.mjtObj, label: str, name: str
    ) -> int:
        object_id = mujoco.mj_name2id(model, object_type, name)
        if object_id < 0:
            raise ModelValidationError(f"missing {label}: {name}")
        return object_id

    def require_site(self, name: str) -> int:
        return self._require_id(self.model, mujoco.mjtObj.mjOBJ_SITE, "site", name)

    def require_body(self, name: str) -> int:
        return self._require_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "body", name)

    def require_geom(self, name: str) -> int:
        return self._require_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom", name)

    def require_sensor(self, name: str) -> int:
        return self._require_id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "sensor", name)

    def validate_actuator_contract(self) -> None:
        for names, arm in ((LEFT_ARM, self.left), (RIGHT_ARM, self.right)):
            for name, joint_id, actuator_id in zip(
                names.actuators, arm.joint_ids, arm.actuator_ids, strict=True
            ):
                self._validate_position_actuator(
                    name, joint_id, actuator_id, exact_control_range=True
                )
        for name, joint_id, actuator_id in zip(
            HAND_ACTUATORS,
            self.hand.joint_ids,
            self.hand.actuator_ids,
            strict=True,
        ):
            self._validate_position_actuator(
                name, joint_id, actuator_id, exact_control_range=False
            )

    def _validate_position_actuator(
        self,
        name: str,
        joint_id: int,
        actuator_id: int,
        *,
        exact_control_range: bool,
    ) -> None:
        if (
            self.model.actuator_trntype[actuator_id]
            != mujoco.mjtTrn.mjTRN_JOINT
            or self.model.actuator_trnid[actuator_id, 0] != joint_id
        ):
            raise ModelValidationError(f"invalid transmission for actuator {name}")
        expected_gear = np.zeros(self.model.actuator_gear.shape[1])
        expected_gear[0] = 1.0
        if not np.array_equal(self.model.actuator_gear[actuator_id], expected_gear):
            raise ModelValidationError(f"actuator must use unit gear: {name}")
        if (
            self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE
            or not self.model.jnt_limited[joint_id]
        ):
            raise ModelValidationError(f"invalid scalar ranged joint for {name}")
        gain = float(self.model.actuator_gainprm[actuator_id, 0])
        bias = self.model.actuator_biasprm[actuator_id]
        if (
            self.model.actuator_gaintype[actuator_id]
            != mujoco.mjtGain.mjGAIN_FIXED
            or self.model.actuator_biastype[actuator_id]
            != mujoco.mjtBias.mjBIAS_AFFINE
            or gain <= 0.0
            or not np.isclose(bias[1], -gain)
            or bias[2] >= 0.0
        ):
            raise ModelValidationError(f"actuator is not position servo: {name}")
        control_range = self.model.actuator_ctrlrange[actuator_id]
        joint_range = self.model.jnt_range[joint_id]
        control_range_valid = (
            np.array_equal(control_range, joint_range)
            if exact_control_range
            else (
                control_range[0] >= joint_range[0]
                and control_range[1] <= joint_range[1]
            )
        )
        if (
            not self.model.actuator_ctrllimited[actuator_id]
            or not control_range_valid
        ):
            raise ModelValidationError(f"invalid control range for {name}")
        if (
            not self.model.actuator_forcelimited[actuator_id]
            or not self.model.jnt_actfrclimited[joint_id]
            or not np.array_equal(
                self.model.actuator_forcerange[actuator_id],
                self.model.jnt_actfrcrange[joint_id],
            )
        ):
            raise ModelValidationError(f"invalid force range for {name}")
