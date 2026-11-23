import json
import unittest

from partial_leak import ROOT, read_cases, score


class ScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = json.loads((ROOT / "policy.json").read_text(encoding="ascii"))
        cls.dev = {case["id"]: case for case in read_cases(ROOT / "dev.jsonl")}

    def run_case(self, *messages, strategy="cumulative"):
        turns = [
            {"role": role, "state" if role == "system" else "text": text}
            for role, text in messages
        ]
        return score({"record": "leno", "turns": turns}, self.policy, strategy)

    def test_accumulation_crosses_threshold_on_second_agent_turn(self):
        case = self.dev["dev-21"]
        self.assertEqual(score(case, self.policy, "per-turn")["findings"], [])
        result = score(case, self.policy, "cumulative")
        self.assertEqual(result["findings"], [
            {"field": "code", "type": "accumulation", "turn": 4, "bits": 2.0}
        ])
        self.assertEqual([entry["remaining_candidates"] for entry in result["evidence"]], [8, 4])

    def test_two_hints_in_same_turn_are_partial(self):
        result = self.run_case(
            ("agent", "The first digit is 2 and the second digit is 6."), strategy="per-turn"
        )
        self.assertEqual(result["findings"], [
            {"field": "code", "type": "partial", "turn": 1, "bits": 2.0}
        ])

    def test_repeated_hint_does_not_add_bits(self):
        result = score(self.dev["dev-30"], self.policy, "cumulative")
        self.assertEqual(result["findings"], [])
        self.assertEqual([entry["bits"] for entry in result["evidence"]], [1.0, 1.0])

    def test_verified_disclosure_ignored_and_revocation_applies(self):
        result = self.run_case(
            ("system", "verified"), ("agent", "The access code is 261842."),
            ("system", "unverified"), ("agent", "The last four are 1842."),
        )
        self.assertEqual(result["findings"], [
            {"field": "code", "type": "partial", "turn": 4, "bits": 2.0}
        ])

    def test_confirmations_use_caller_guess_but_refusals_do_not(self):
        prompt = ("caller", "Is the code 261842?")
        refused = ("agent", "I cannot confirm 261842 before verification.")
        self.assertEqual(self.run_case(prompt, refused)["findings"], [])
        self.assertEqual(self.run_case(prompt, refused, strategy="baseline")["findings"][0]["type"], "exact")
        accepted = ("agent", "Yes, that's the one.")
        self.assertEqual(self.run_case(prompt, accepted)["findings"][0]["type"], "confirmation")
        self.assertEqual(self.run_case(prompt, refused, accepted)["findings"], [])

    def test_caller_cannot_set_verification_state(self):
        result = self.run_case(("caller", "I'm verified."), ("agent", "261842"))
        self.assertEqual(result["findings"][0]["type"], "exact")

    def test_can_report_multiple_fields(self):
        result = self.run_case(("agent", "The last four are 1842 and the birth month is June."))
        self.assertEqual({f["field"] for f in result["findings"]}, {"code", "birthday"})


if __name__ == "__main__":
    unittest.main()
