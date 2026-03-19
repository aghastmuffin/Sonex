import unittest
from unittest.mock import patch

import Sonex


class DistributionEntrypointTests(unittest.TestCase):
    @patch("Sonex.subprocess.run")
    def test_main_runs_gui_and_frase_when_done(self, mock_run):
        mock_run.side_effect = [
            type("Proc", (), {"stdout": "Done"})(),
            type("Proc", (), {"stdout": ""})(),
        ]

        Sonex.main()

        self.assertEqual(mock_run.call_count, 2)
        first_args = mock_run.call_args_list[0].args[0]
        second_args = mock_run.call_args_list[1].args[0]
        self.assertEqual(first_args[1:], ["-m", "ui.gui"])
        self.assertEqual(second_args[1:], ["-m", "ui.frase"])


if __name__ == "__main__":
    unittest.main()
