import os
from pathlib import Path
import tempfile
import unittest

from inference import check_file


class CheckFileTests(unittest.TestCase):
    def test_valid_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(b"content")
            temp_path = temp.name

        try:
            resolved = check_file(temp_path)
            self.assertEqual(resolved, str(Path(temp_path).resolve()))
        finally:
            os.remove(temp_path)

    def test_file_not_found(self):
        non_existent_path = "/path/to/non_existent_file_12345.txt"
        with self.assertRaises(ValueError) as ctx:
            check_file(non_existent_path)
        self.assertIn("File not found:", str(ctx.exception))

    def test_directory_instead_of_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError) as ctx:
                check_file(temp_dir)
            self.assertIn("File not found:", str(ctx.exception))

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = temp.name

        try:
            with self.assertRaises(ValueError) as ctx:
                check_file(temp_path)
            self.assertIn("Empty file:", str(ctx.exception))
        finally:
            os.remove(temp_path)

    def test_expand_user(self):
        home_dir = Path.home()
        test_file = home_dir / f".test_check_file_{os.getpid()}.tmp"
        test_file.write_bytes(b"data")

        try:
            relative_home_path = f"~/{test_file.name}"
            resolved = check_file(relative_home_path)
            self.assertEqual(resolved, str(test_file.resolve()))
        finally:
            if test_file.exists():
                test_file.unlink()


if __name__ == "__main__":
    unittest.main()
