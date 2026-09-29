from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from mods.jevai import package


class PackageOutputTests(unittest.TestCase):
    def test_reject_roots_and_unmarked_directories(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(package, 'ROOT', folder):
            root=Path(folder)
            for p in [root, root/'temp', root/'temp/release', root/'temp/dataset/records']:
                with self.assertRaises(ValueError):package.output_directory(str(p))
            out=root/'temp/release-mp/jevai'
            self.assertEqual(package.output_directory(str(out)),str(out.resolve()))
            out.mkdir(parents=True);(out/'user-data.txt').write_text('keep')
            with self.assertRaises(ValueError):package.output_directory(str(out))
            (out/'descriptor.mod').write_text('name="JevAI"')
            self.assertEqual(package.output_directory(str(out)),str(out.resolve()))


if __name__=='__main__':unittest.main()
