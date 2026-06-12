import unittest

from src.deploy.palette import CLASS_NAMES_9, PALETTE_9, get_class_set, palette_for_class_names


class Palette9ClassTests(unittest.TestCase):
    def test_9class_order(self):
        self.assertEqual(
            CLASS_NAMES_9,
            ["background", "skin", "l_brow", "r_brow", "mouth", "u_lip", "l_lip", "teeth", "hair"],
        )
        self.assertEqual(len(PALETTE_9), 9)

    def test_get_class_set(self):
        class_names, palette = get_class_set("mouth2teeth_9cls")
        self.assertEqual(class_names, CLASS_NAMES_9)
        self.assertEqual(len(palette), 9)

    def test_palette_for_class_names(self):
        palette = palette_for_class_names(["background", "teeth", "hair"])
        self.assertEqual(palette[0], [0, 0, 0])
        self.assertEqual(palette[1], [0, 255, 255])
        self.assertEqual(palette[2], [255, 255, 170])


if __name__ == "__main__":
    unittest.main()
