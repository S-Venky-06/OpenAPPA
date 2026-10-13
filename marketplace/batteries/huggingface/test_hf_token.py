import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
import contextlib
from unittest.mock import patch


MODULE = Path(__file__).with_name("hf_token.py")
SPEC = importlib.util.spec_from_file_location("hf_token", MODULE)
HF_TOKEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HF_TOKEN)


class ResolveToken(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)

    def stored(self, token, path=None):
        path = path or Path(self.home.name) / ".cache" / "huggingface" / "token"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token)
        return path

    def test_the_variable_wins_over_the_stored_login(self):
        self.stored("hf_stored")
        token = HF_TOKEN.resolve_token({"HOME": self.home.name, "APPA_PROVIDER_HUGGINGFACE_TOKEN": " hf_var\n"})
        self.assertEqual(token, "hf_var")

    def test_the_stored_login_answers_when_the_variable_is_blank_or_unset(self):
        self.stored("hf_stored\n")
        self.assertEqual(HF_TOKEN.resolve_token({"HOME": self.home.name}), "hf_stored")
        self.assertEqual(HF_TOKEN.resolve_token({"HOME": self.home.name, "APPA_PROVIDER_HUGGINGFACE_TOKEN": "  "}), "hf_stored")

    def test_hf_home_and_hf_token_path_relocate_the_stored_login(self):
        home = self.stored("from_hf_home", Path(self.home.name) / "hfhome" / "token")
        self.assertEqual(HF_TOKEN.resolve_token({"HF_HOME": str(home.parent)}), "from_hf_home")
        explicit = self.stored("from_path", Path(self.home.name) / "elsewhere")
        self.assertEqual(HF_TOKEN.resolve_token({"HF_HOME": str(home.parent), "HF_TOKEN_PATH": str(explicit)}), "from_path")

    def test_xdg_cache_home_relocates_the_default_login(self):
        cache = Path(self.home.name) / "xdg"
        self.stored("from_xdg", cache / "huggingface" / "token")
        self.assertEqual(HF_TOKEN.resolve_token({"HOME": self.home.name, "XDG_CACHE_HOME": str(cache)}), "from_xdg")

    def test_neither_present_names_both_fixes(self):
        self.stored("")
        with self.assertRaises(RuntimeError) as refused:
            HF_TOKEN.resolve_token({"HOME": self.home.name})
        self.assertIn("APPA_PROVIDER_HUGGINGFACE_TOKEN", str(refused.exception))
        self.assertIn("hf auth login", str(refused.exception))
        with self.assertRaises(RuntimeError):
            HF_TOKEN.resolve_token({})

    def test_a_multi_user_deployment_reads_the_variable_alone(self):
        self.stored("hf_stored")
        environ = {"HOME": self.home.name}
        self.assertEqual(HF_TOKEN.Deployment.SINGLE_USER.token(environ), "hf_stored")
        with self.assertRaises(RuntimeError):
            HF_TOKEN.Deployment.MULTI_USER.token(environ)
        with self.assertRaises(RuntimeError):
            HF_TOKEN.env_token({**environ, "APPA_PROVIDER_HUGGINGFACE_TOKEN": "  "})
        self.assertEqual(HF_TOKEN.Deployment.MULTI_USER.token({**environ, "APPA_PROVIDER_HUGGINGFACE_TOKEN": " hf_var\n"}), "hf_var")

    def test_the_hub_root_is_hf_endpoint(self):
        self.assertEqual(HF_TOKEN.hub_root({}), "https://huggingface.co")
        self.assertEqual(HF_TOKEN.hub_root({"HF_ENDPOINT": "http://127.0.0.1:9/"}), "http://127.0.0.1:9")

    def test_tilde_expansion_in_paths(self):
        self.stored("from_token_path")

        token = HF_TOKEN.resolve_token({
            "HOME": self.home.name,
            "HF_TOKEN_PATH": "~/.cache/huggingface/token"
        })
        self.assertEqual(token, "from_token_path")

        token = HF_TOKEN.resolve_token({
            "HOME": self.home.name,
            "HF_HOME": "~/.cache/huggingface"
        })
        self.assertEqual(token, "from_token_path")

        token = HF_TOKEN.resolve_token({
            "HOME": self.home.name,
            "XDG_CACHE_HOME": "~/.cache"
        })
        self.assertEqual(token, "from_token_path")

    def test_unresolved_explicit_tilde_remains_literal(self):
        with contextlib.chdir(self.home.name):
            with self.assertRaises(RuntimeError) as exc:
                HF_TOKEN.resolve_token({
                    "HF_TOKEN_PATH": "~/token"
                })
            self.assertIn("~/token", str(exc.exception).replace("\\", "/"))

    def test_unchanged_relative_path(self):
        with self.assertRaises(RuntimeError) as exc:
            HF_TOKEN.resolve_token({
                "HF_TOKEN_PATH": "./relative/token"
            })
        self.assertIn("relative/token", str(exc.exception).replace("\\", "/"))

    def test_empty_home_must_not_load_working_directory_token(self):
        with contextlib.chdir(self.home.name):
            fake_token_path = Path(".cache") / "huggingface" / "token"
            fake_token_path.parent.mkdir(parents=True, exist_ok=True)
            fake_token_path.write_text("cwd_token")

            with self.assertRaises(RuntimeError) as exc:
                HF_TOKEN.resolve_token({"HOME": ""})
            self.assertIn("APPA_PROVIDER_HUGGINGFACE_TOKEN is not set", str(exc.exception))


class ExpandTildeTests(unittest.TestCase):
    def check_expand(self, os_name, environ, path, expected):
        with patch.object(HF_TOKEN.os, "name", os_name):
            self.assertEqual(HF_TOKEN._expand_tilde(path, environ), expected)

    def test_windows_precedence(self):
        cases = [
            ({"USERPROFILE": "C:\\Users\\profile", "HOMEDRIVE": "D:", "HOMEPATH": "\\Users\\home", "HOME": "E:\\home"}, "~", "C:\\Users\\profile"),
            ({"HOMEDRIVE": "D:", "HOMEPATH": "\\Users\\home", "HOME": "E:\\home"}, "~", "D:\\Users\\home"),
            ({"HOME": "E:\\home"}, "~", "E:\\home"),
            ({"USERPROFILE": "", "HOME": "E:\\home"}, "~", "E:\\home"),
            ({"HOMEDRIVE": "", "HOMEPATH": "\\Users\\test", "HOME": "E:\\home"}, "~", "E:\\home"),
            ({"HOMEDRIVE": "C:", "HOMEPATH": "", "HOME": "E:\\home"}, "~", "E:\\home"),
            ({"USERPROFILE": "", "HOME": ""}, "~/foo", "~/foo"),
            ({"USERPROFILE": "C:\\Users\\test"}, "~/foo", "C:\\Users\\test/foo"),
            ({"USERPROFILE": "C:\\Users\\test"}, "~\\foo", "C:\\Users\\test\\foo"),
        ]
        for env, path, expected in cases:
            with self.subTest(env=env, path=path):
                self.check_expand("nt", env, path, expected)

    def test_unix_precedence(self):
        cases = [
            ({"HOME": "/home/test"}, "~", "/home/test"),
            ({"USERPROFILE": "/home/windows", "HOME": "/home/test"}, "~", "/home/test"),
            ({"HOME": ""}, "~/foo", "~/foo"),
            ({}, "~/foo", "~/foo"),
            ({"HOME": "/home/test"}, "~\\foo", "~\\foo"),
        ]
        for env, path, expected in cases:
            with self.subTest(env=env, path=path):
                self.check_expand("posix", env, path, expected)

    def test_shared_behavior(self):
        cases = [
            ({"HOME": "/home/test"}, "/absolute/path", "/absolute/path"),
            ({"HOME": "/home/test"}, "relative/path", "relative/path"),
            ({"HOME": "/home/test"}, "~", "/home/test"),
            ({"HOME": "/home/test"}, "~username/foo", "~username/foo"),
            ({"HOME": "/home/ test "}, "~", "/home/ test "),
            ({"HOME": " "}, "~", " "),
        ]
        for os_name in ("nt", "posix"):
            for env, path, expected in cases:
                with self.subTest(os_name=os_name, env=env, path=path):
                    self.check_expand(os_name, env, path, expected)


class HomeDirTests(unittest.TestCase):
    """home_dir returns the correct directory for both platforms."""

    def test_posix_uses_home(self):
        self.assertEqual(HF_TOKEN.home_dir({"HOME": "/home/user"}, windows=False), "/home/user")

    def test_posix_returns_none_when_home_unset(self):
        self.assertIsNone(HF_TOKEN.home_dir({}, windows=False))

    def test_posix_returns_none_when_home_empty(self):
        self.assertIsNone(HF_TOKEN.home_dir({"HOME": ""}, windows=False))

    def test_windows_prefers_userprofile(self):
        env = {"USERPROFILE": r"C:\Users\user", "HOMEDRIVE": "C:", "HOMEPATH": r"\Users\other"}
        self.assertEqual(HF_TOKEN.home_dir(env, windows=True), r"C:\Users\user")

    def test_windows_falls_back_to_homedrive_homepath(self):
        env = {"HOMEDRIVE": "C:", "HOMEPATH": r"\Users\user"}
        self.assertEqual(HF_TOKEN.home_dir(env, windows=True), r"C:\Users\user")

    def test_windows_falls_back_to_home(self):
        env = {"HOME": r"C:\Users\user"}
        self.assertEqual(HF_TOKEN.home_dir(env, windows=True), r"C:\Users\user")

    def test_windows_returns_none_when_nothing_set(self):
        self.assertIsNone(HF_TOKEN.home_dir({}, windows=True))


class ExpandTildeTests(unittest.TestCase):
    """_expand_tilde works correctly on both platforms without mocking os.name."""

    def test_posix_expands_tilde(self):
        result = HF_TOKEN._expand_tilde("~/.cache/hf/token", {"HOME": "/home/user"}, windows=False)
        self.assertEqual(result, "/home/user/.cache/hf/token")

    def test_posix_leaves_tilde_when_no_home(self):
        result = HF_TOKEN._expand_tilde("~/file", {}, windows=False)
        self.assertEqual(result, "~/file")

    def test_windows_expands_tilde_via_userprofile(self):
        env = {"USERPROFILE": r"C:\Users\user"}
        result = HF_TOKEN._expand_tilde(r"~\.cache\hf\token", env, windows=True)
        self.assertEqual(result, r"C:\Users\user\.cache\hf\token")

    def test_windows_expands_tilde_via_homedrive_homepath(self):
        env = {"HOMEDRIVE": "C:", "HOMEPATH": r"\Users\user"}
        result = HF_TOKEN._expand_tilde("~/file", env, windows=True)
        self.assertEqual(result, r"C:\Users\user/file")

    def test_windows_leaves_tilde_when_nothing_set(self):
        result = HF_TOKEN._expand_tilde("~/file", {}, windows=True)
        self.assertEqual(result, "~/file")

    def test_non_tilde_path_is_unchanged(self):
        result = HF_TOKEN._expand_tilde("/absolute/path", {"HOME": "/home/user"}, windows=False)
        self.assertEqual(result, "/absolute/path")


class TokenPathWindowsTests(unittest.TestCase):
    """token_path resolves Windows home variables correctly without mocking."""

    def setUp(self):
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)

    def test_resolves_via_userprofile_on_windows(self):
        env = {"USERPROFILE": self.home.name}
        expected = Path(self.home.name) / ".cache" / "huggingface" / "token"
        self.assertEqual(HF_TOKEN.token_path(env, windows=True), expected)

    def test_resolves_via_homedrive_homepath_on_windows(self):
        # Split the temp path string manually to avoid pathlib's relative_to("C:") quirk.
        home_str = self.home.name
        # Find the drive separator: "C:\..." -> drive="C:", rest="\..."
        colon_idx = home_str.find(":")
        if colon_idx != -1:
            drive = home_str[: colon_idx + 1]
            rest = home_str[colon_idx + 1 :]
        else:
            drive = ""
            rest = home_str
        env = {"HOMEDRIVE": drive, "HOMEPATH": rest}
        expected = Path(home_str) / ".cache" / "huggingface" / "token"
        self.assertEqual(HF_TOKEN.token_path(env, windows=True), expected)

    def test_resolves_via_home_on_posix(self):
        env = {"HOME": self.home.name}
        expected = Path(self.home.name) / ".cache" / "huggingface" / "token"
        self.assertEqual(HF_TOKEN.token_path(env, windows=False), expected)

    def test_hf_token_path_takes_precedence_regardless_of_platform(self):
        explicit = str(Path(self.home.name) / "my_token")
        for windows in (True, False):
            with self.subTest(windows=windows):
                self.assertEqual(
                    HF_TOKEN.token_path({"HF_TOKEN_PATH": explicit}, windows=windows),
                    Path(explicit),
                )


if __name__ == "__main__":
    unittest.main()
