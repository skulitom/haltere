"""Read a Liftoff ``.drone`` configuration (XML) and derive controller/rate parameters for the simulator."""
from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from .controller import RateControllerParams

LIFTOFF_LOCALLOW = os.path.join(os.path.expanduser('~'), 'AppData', 'LocalLow', 'LuGus Studios', 'Liftoff')
XSI_TYPE = '{http://www.w3.org/2001/XMLSchema-instance}type'


@dataclass
class LiftoffDroneConfig:
    name: str
    controller: str
    pid: dict            # axis -> (P, I, D)
    rates: dict          # axis -> (rc_rate, expo, super_rate) as fractions
    idle: float
    mix: list[list[float]]
    path: str

    def rate_params(self) -> RateControllerParams:
        P = tuple(self.pid[a][0] for a in ('roll', 'pitch', 'yaw'))
        I = tuple(self.pid[a][1] for a in ('roll', 'pitch', 'yaw'))
        D = tuple(self.pid[a][2] for a in ('roll', 'pitch', 'yaw'))
        return RateControllerParams(P=P, I=I, D=D, idle=self.idle, mix=self.mix)


def list_drone_configs(root: str = LIFTOFF_LOCALLOW) -> list[str]:
    return sorted(glob.glob(os.path.join(root, 'DroneConfigurations', '*', '*.drone')))


def _floats(elem) -> list[float]:
    return [float(x.text) for x in elem.findall('float')]


def parse_drone_config(path: str) -> LiftoffDroneConfig:
    root = ET.parse(path).getroot()
    name = root.findtext('name', default=os.path.basename(path))
    selected = root.findtext('SelectedController', default='ZETAFLIGHT')
    fcs = root.find('FlightControllers')
    zeta = beta = None
    for s in ([] if fcs is None else fcs.findall('Settings')):
        t = s.get(XSI_TYPE, '')
        if t == 'ZetaFlightControllerSettings':
            zeta = s
        elif t == 'BetaFlightControllerSettings':
            beta = s
    pid: dict = {}
    rates: dict = {}
    idle = 0.04
    mix = [[1, 1, 1, -1], [1, -1, 1, 1], [1, 1, -1, 1], [1, -1, -1, -1]]
    if beta is not None:
        tbl = beta.find('motorMixingTable')
        if tbl is not None:
            mix = [_floats(row) for row in tbl.findall('ArrayOfFloat')]
    if selected.upper().startswith('ZETA') and zeta is not None:
        gains = zeta.find('PIDControllerSettings/PIDGains')
        rows = gains.findall('PIDGains') if gains is not None else []
        for axis, row in zip(('roll', 'pitch', 'yaw'), rows):
            pid[axis] = (float(row.findtext('P_Gain')), float(row.findtext('I_Gain')), float(row.findtext('D_Gain')))
        am = zeta.find('InputModifierSettings/AxisModels/AxisModel')
        if am is not None:
            for axis in ('Roll', 'Pitch', 'Yaw'):
                e = am.find(axis)
                rates[axis.lower()] = (float(e.findtext('Rate')) / 100, float(e.findtext('Expo')) / 100,
                                       float(e.findtext('SuperExpo')) / 100)
        idle = float(zeta.findtext('idleMotorThrottlePercentage', default='0.04'))
    elif beta is not None:
        for axis, tag in (('roll', 'pidRollRate'), ('pitch', 'pidPitchRate'), ('yaw', 'pidYawRate')):
            e = beta.find(tag)
            pid[axis] = (float(e.findtext('_kP')), float(e.findtext('_kI')), float(e.findtext('_kD')))
        for axis in ('roll', 'pitch', 'yaw'):
            rates[axis] = (float(beta.findtext(f'{axis}Rate')) / 100, float(beta.findtext(f'{axis}Expo')) / 100,
                           float(beta.findtext(f'{axis}SuperExpo')) / 100)
        idle = (float(beta.findtext('idleMotorThrottle', default='1040')) - 1000) / 1000
    if not pid:
        pid = {'roll': (29, 34, 22), 'pitch': (29, 36, 22), 'yaw': (30, 38, 0)}
    if not rates:
        rates = {a: (1.0, 0.0, 0.7) for a in ('roll', 'pitch', 'yaw')}
    return LiftoffDroneConfig(name=name, controller=selected, pid=pid, rates=rates, idle=idle, mix=mix, path=path)
