import pathlib
import subprocess
import unittest


LIB_PATH = pathlib.Path(__file__).resolve().parents[1] / 'shell_quote_lib.sh'


class ShellQuoteLibTest(unittest.TestCase):
    def roundtrip(self, value: str) -> str:
        result = subprocess.run(
            [
                'bash',
                '-lc',
                f'source {LIB_PATH}; quoted=$(shell_single_quote "$VALUE"); eval "printf %s $quoted"',
            ],
            check=False,
            capture_output=True,
            text=True,
            env={'VALUE': value},
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        return result.stdout

    def test_roundtrip_plain_string(self):
        self.assertEqual(self.roundtrip('plain text'), 'plain text')

    def test_roundtrip_string_with_single_quotes(self):
        prompt = "Question: What's on the dog's collar?"
        self.assertEqual(self.roundtrip(prompt), prompt)


if __name__ == '__main__':
    unittest.main()
