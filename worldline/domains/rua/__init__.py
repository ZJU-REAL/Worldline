"""The first Worldline domain: robot use through observation/action interfaces."""

from worldline.plugins import EnvironmentPlugin


def plugins() -> tuple[EnvironmentPlugin, ...]:
    from worldline.adapters.rua.socket import RUASocketEnvironment, managed_session
    from worldline.domains.rua.mujoco_reach import MujocoReach

    return (
        EnvironmentPlugin("rua.mujoco-reach", "rua", MujocoReach.version(), MujocoReach),
        EnvironmentPlugin("rua.socket", "rua", "rua-robot/1.0-adapter/0.2", RUASocketEnvironment, restartable=False),
        EnvironmentPlugin("rua.managed", "rua", "rua-robot/1.0-managed/0.1", managed_session),
    )
