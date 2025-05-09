''' test global - parametric consistency of point mass and drone models '''
import unittest

import numpy as np

from drone3d.centerlines.spline_centerline import SplineCenterline, SplineCenterlineConfig, \
    SplineRyFitOptions
from drone3d.dynamics.point_model import PointConfig, PointModel, ParametricPointModel
from drone3d.dynamics.drone_models import DroneConfig, DroneModel, ParametricDroneModel

DIST_TOL =1e-4

class TestKinematics(unittest.TestCase):
    ''' kinematics tests'''

    def _run_pmm_test(self, global_r: bool, planar_fit: bool):
        ''' forwards motion of point mass, should match global motion '''
        print(f'PMM Test: Global R: {global_r}, planar_fit: {planar_fit}', end='')
        x = np.array([0, 10, 0])
        y = np.array([0, 10, 20])
        z = np.array([0, 5, 10])

        config = SplineCenterlineConfig(x = np.array([x, y, z]))
        config.closed = False
        if planar_fit:
            config.ry_fit_method = SplineRyFitOptions.PLANAR
        else:
            config.ry_fit_method = SplineRyFitOptions.TORSION_FREE

        cent = SplineCenterline(config)

        point_config = PointConfig(g = 1, global_r=global_r)
        model = ParametricPointModel(point_config, cent)

        global_model = PointModel(point_config)

        state = model.get_empty_state()
        global_state = global_model.get_empty_state()
        state.v.v1 = 1
        global_state.v.from_vec(cent.p2es(0))
        if point_config.global_r:
            state.v.from_vec(cent.p2es(0))

        for _ in range(100):
            global_model.step(global_state)
            model.step(state)
            self.assertTrue(np.linalg.norm(state.x.to_vec() - global_state.x.to_vec()) < DIST_TOL)

        final_err = np.linalg.norm(state.x.to_vec() - global_state.x.to_vec())
        print(f' \t Final position error: error: {final_err:0.6f}m')

    def test_pmm_kinematics(self):
        ''' test point mass kinematics'''
        print('')
        for global_r in [True, False]:
            for planar_fit in [True, False]:
                self._run_pmm_test(global_r=global_r, planar_fit=planar_fit)

    def _run_drone_test(self, global_r: bool, planar_fit: bool, use_esp: bool):
        ''' forwards motion of point mass, should match global motion '''
        print(f'Drone Test: Global R: {global_r}, planar_fit: {planar_fit}, ESP: {use_esp}',
              end = '')
        x = np.array([0, 10, 0])
        y = np.array([0, 10, 20])
        z = np.array([0, 5, 10])

        config = SplineCenterlineConfig(x = np.array([x, y, z]))
        config.closed = False
        if planar_fit:
            config.ry_fit_method = SplineRyFitOptions.PLANAR
        else:
            config.ry_fit_method = SplineRyFitOptions.TORSION_FREE

        cent = SplineCenterline(config)

        drone_config = DroneConfig(g = 1, global_r=global_r, use_quat=use_esp)
        model = ParametricDroneModel(drone_config, cent)

        global_model = DroneModel(drone_config)

        state = model.get_empty_state()
        global_state = global_model.get_empty_state()
        state.v.v1 = 1
        global_state.v.from_vec(cent.p2es(0))
        if drone_config.global_r:
            state.v.from_vec(cent.p2es(0))

        for _ in range(100):
            global_model.step(global_state)
            model.step(state)
            self.assertTrue(np.linalg.norm(state.x.to_vec() - global_state.x.to_vec()) < DIST_TOL)

        final_err = np.linalg.norm(state.x.to_vec() - global_state.x.to_vec())
        print(f' \t Final position error: error: {final_err:0.6f}m')

    def test_drone_kinematics(self):
        ''' test drone kinematics'''
        print('')
        for global_r in [True, False]:
            for planar_fit in [True, False]:
                for use_esp in [True, False]:
                    self._run_drone_test(global_r=global_r, planar_fit=planar_fit, use_esp=use_esp)

if __name__ == '__main__':
    unittest.main(verbosity=2)
