import unittest

from fm_humanoid_bench.adapters import BRAIN_ADAPTERS, BODY_ADAPTERS, require_body, require_brain, support_matrix


class AdapterRegistryTests(unittest.TestCase):
    def test_all_declared_brains_share_reference40(self):
        self.assertEqual(set(BRAIN_ADAPTERS), {"gpt6", "psi0", "groot", "pi05", "vla_jepa", "dit4dit"})
        self.assertEqual({entry["output"] for entry in BRAIN_ADAPTERS.values()}, {"unitree_g1_gmt_refpose_v3_1"})

    def test_all_five_bodies_support_both_tracks(self):
        self.assertEqual(set(BODY_ADAPTERS), {"sonic", "twist2", "scale", "holo", "zero"})
        for name in BODY_ADAPTERS:
            self.assertTrue(require_body(name, "reference")["reference"])
            self.assertTrue(require_body(name, "native")["native"])

    def test_catalog_transport_mismatch_is_rejected(self):
        require_brain({"family": "psi0", "kind": "http"})
        require_brain({"family": "gpt6", "kind": "file"})
        with self.assertRaises(ValueError):
            require_brain({"family": "dit4dit", "kind": "file"})
        self.assertEqual(len(support_matrix()["bodies"]), 5)


if __name__ == "__main__":
    unittest.main()
