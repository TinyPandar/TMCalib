import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ProfileArchitectureTests(unittest.TestCase):
    def test_roi26_entrypoint_is_only_a_profile_wrapper(self):
        path = ROOT / "calibrate_128x128_26x26.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        class_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }

        self.assertLess(len(source.splitlines()), 80)
        self.assertEqual(class_names, {"Application"})
        self.assertIn("from calibrate_128x128 import Application", source)
        self.assertNotIn("class CameraHandler", source)
        self.assertNotIn("class DMDController", source)

    def test_shared_core_accepts_profile_camera_configuration(self):
        # This source-level contract is intentionally checked without importing
        # PySpin or loading the JUOPT DLL on CI/developer machines.
        source = (ROOT / "calibrate_128x128.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="calibrate_128x128.py")
        camera_init = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "CameraHandler":
                camera_init = next(
                    item
                    for item in node.body
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__"
                )
                break

        self.assertIsNotNone(camera_init)
        parameters = {argument.arg for argument in camera_init.args.args}
        self.assertTrue(
            {"roi_shape", "polarization_channel", "exposure_us", "output_tag"}
            .issubset(parameters)
        )


if __name__ == "__main__":
    unittest.main()
