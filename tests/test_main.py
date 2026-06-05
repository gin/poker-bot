import json
import os
import tempfile
import unittest

from main import load_credentials, make_api_client


class DummyResult:
    def __init__(self, stdout):
        self.stdout = stdout


def fake_runner(cmd, capture_output, text, timeout):
    return DummyResult('{"success": true}')


class PokerBotTests(unittest.TestCase):
    def test_load_credentials_json(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            json.dump(
                {
                    "apiKey": "arena_sk_abc",
                    "competitionId": "cmpy_1",
                    "agentId": "cmpy_agent",
                },
                tmp,
            )
            tmp_path = tmp.name

        api_key, competition_id, agent_id = load_credentials(tmp_path)
        self.assertEqual(api_key, "arena_sk_abc")
        self.assertEqual(competition_id, "cmpy_1")
        self.assertEqual(agent_id, "cmpy_agent")
        os.unlink(tmp_path)

    def test_load_credentials_env_style(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write("apiKey=arena_sk_xyz\n")
            tmp.write("competitionId=cmpy_2\n")
            tmp.write("agentId=cmpy_agent2\n")
            tmp_path = tmp.name

        api_key, competition_id, agent_id = load_credentials(tmp_path)
        self.assertEqual(api_key, "arena_sk_xyz")
        self.assertEqual(competition_id, "cmpy_2")
        self.assertEqual(agent_id, "cmpy_agent2")
        os.unlink(tmp_path)

    def test_api_runner_injection(self):
        api_fn = make_api_client("arena_sk_test", runner=fake_runner)
        result = api_fn("GET", "/test")
        self.assertEqual(result, {"success": True})


if __name__ == "__main__":
    unittest.main()
